/*
* To change this template, choose Tools | Templates
* and open the template in the editor.
*/
package tests;

import java.io.StringWriter;
import java.io.Writer;
import java.nio.IntBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;

import ai.PassiveAI;
import ai.RandomBiasedAI;
import ai.RandomNoAttackAI;
import ai.core.AI;
import ai.jni.JNIAI;
import ai.rewardfunction.RewardFunctionInterface;
import ai.jni.JNIInterface;
import ai.jni.Response;
import ai.jni.Responses;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.Trace;
import rts.TraceEntry;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitTypeTable;
import util.NDBuffer;
import tests.JNIGridnetClientSelfPlay;

/**
 *
 * Improved performance for JVM <-> NumPy data exchange
 * with direct buffer (JVM allocated).
 * 
 */
public class JNIGridnetSharedMemVecClient {

    public static final int ACTION_DIM = 7;

    public final JNIGridnetSharedMemClient[] clients;
    public final JNIGridnetSharedMemClientSelfPlay[] selfPlayClients;
    public final int maxSteps;
    public final int[] envSteps; 
    public final RewardFunctionInterface[] rfs;
    final int numRfs;
    public final UnitTypeTable utt;
    public final boolean partialObs;
    public final String mapPath;

    // storage
    final NDBuffer obsBuffer;
    final NDBuffer actionMaskBuffer;
    final NDBuffer actionBuffer;
    final double[][] reward;
    final boolean[][] done;
    final Response[] rs;
    final Responses responses;
    final double[][] terminalReward;
    final boolean[][] terminalDone;

    // synchronization primitives
    final ArrayList<Thread> pool;
    final AtomicRefernce<CountDownLatch> clientStepBlocker;

    public JNIGridnetSharedMemVecClient(int a_num_selfplayenvs, int a_num_envs, int a_max_steps, RewardFunctionInterface[] a_rfs,
            String a_micrortsPath, String mapPath, AI[] a_ai2s, UnitTypeTable a_utt, boolean partial_obs,
            IntBuffer obsBuffer, IntBuffer actionMaskBuffer, IntBuffer actionBuffer,
            int threadPoolSize) throws Exception {
        maxSteps = a_max_steps;
        utt = a_utt;
        rfs = a_rfs;
        numRfs = a_rfs.length;
        partialObs = partial_obs;

        this.mapPath = Paths.get(a_micrortsPath, mapPath).toString();

        // get dims
        final PhysicalGameState pgs = PhysicalGameState.load(this.mapPath, utt);
        final int s1 = a_num_selfplayenvs + a_num_envs;
        final int maxAttackRadius = utt.getMaxAttackRange() * 2 + 1;
        final int actionMaskNumEntries = 6+4+4+4+4+utt.getUnitTypes().size()+maxAttackRadius*maxAttackRadius;

        // initialize shared storage
        this.obsBuffer = new NDBuffer(obsBuffer, new int[]{s1, pgs.getHeight(), pgs.getWidth(), GameState.numFeaturePlanes});
        this.actionMaskBuffer = new NDBuffer(actionMaskBuffer, new int[]{s1, pgs.getHeight(), pgs.getWidth(), actionMaskNumEntries});
        this.actionBuffer = new NDBuffer(actionBuffer, new int[]{s1, pgs.getHeight() * pgs.getWidth(), ACTION_DIM});

        // initialize clients
        envSteps = new int[a_num_selfplayenvs + a_num_envs];
        final CountDownLatch clientReadyCounter = new CountDownLatch(a_num_selfplayenvs/2 + a_num_envs);
        clientStepBlocker = new AtomicReference<>(new CountDownLatch(1));

        selfPlayClients = new JNIGridnetSharedMemClientSelfPlay[a_num_selfplayenvs/2];
        for (int i = 0; i < selfPlayClients.length; i++) {
            int clientOffset = i*2;
            selfPlayClients[i] = new JNIGridnetSharedMemClientSelfPlay(a_rfs, a_micrortsPath, mapPath, a_utt, partialObs,
                clientOffset, this.obsBuffer, this.actionMaskBuffer, this.actionBuffer,
                clientReadyCounter, clientStepBlocker);
        }
        clients = new JNIGridnetSharedMemClient[a_num_envs];
        for (int i = 0; i < clients.length; i++) {
            int clientOffset = i+selfPlayClients.length*2;
            clients[i] = new JNIGridnetSharedMemClient(a_rfs, this.mapPath, a_ai2s[i], a_utt, partialObs,
                clientOffset, this.obsBuffer, this.actionMaskBuffer, this.actionBuffer,
                clientReadyCounter, clientStepBlocker);
        }

        // initialize storage
        reward = new double[s1][numRfs];
        done = new boolean[s1][numRfs];
        responses = new Responses(null, null, null);
        rs = new Response[s1];
        terminalReward = new double[2][numRfs];
        terminalDone = new boolean[2][numRfs];

        if (threadPoolSize > 0) {
            pool = new ArrayList<>();
            for (int i = 0; i < selfPlayClients.length; i++) {
                pool.add(new Thread(selfPlayClients[i]));
            }
            for (int i = 0; i < clients.length; i++) {
                pool.add(new Thread(clients[i]));
            }

            pool.forEach(Thread::start);
            // wait or all threads to start
            clientReadyCounter.await();
        } else {
            pool = null;
        }
    }

    public Responses reset(int[] players) throws Exception {
        for (int i = 0; i < selfPlayClients.length; i++) {
            selfPlayClients[i].reset();
            rs[i*2] = selfPlayClients[i].getResponse(0);
            rs[i*2+1] = selfPlayClients[i].getResponse(1);
        }
        for (int i = selfPlayClients.length*2; i < players.length; i++) {
            clients[i-selfPlayClients.length*2].reset(players[i]);
            rs[i] = clients[i-selfPlayClients.length*2].getResponse();
        }

        for (int i = 0; i < rs.length; i++) {
            reward[i] = rs[i].reward;
            done[i] = rs[i].done;
        }
        responses.set(null, reward, done);
        return responses;
    }

    public Responses gameStep(int[] players) throws Exception {
        if (pool != null) {
            // swap lock for the next operation
            clientStepBlocker.getAndSet(new CountDownLatch(1)).countDown();
            // wait for all threads to finish
            clientStepReady.wait();
        }

        for (int i = 0; i < selfPlayClients.length; i++) {
            if (null == pool) {
                selfPlayClients[i].gameStep();
            }
            rs[i*2] = selfPlayClients[i].getResponse(0);
            rs[i*2+1] = selfPlayClients[i].getResponse(1);
            envSteps[i*2] += 1;
            envSteps[i*2+1] += 1;
            if (rs[i*2].done[0] || envSteps[i*2] >= maxSteps) {
                // stash previous values
                System.arraycopy(rs[i*2].reward, 0, terminalReward[0], 0, numRfs);
                System.arraycopy(rs[i*2].done, 0, terminalDone[0], 0, numRfs);
                System.arraycopy(rs[i*2+1].reward, 0, terminalReward[1], 0, numRfs);
                System.arraycopy(rs[i*2+1].done, 0, terminalDone[1], 0, numRfs);

                selfPlayClients[i].reset();

                // restore previosly stashed values
                System.arraycopy(terminalReward[0], 0, rs[i*2].reward, 0, numRfs);
                System.arraycopy(terminalDone[0], 0, rs[i*2].done, 0, numRfs);
                System.arraycopy(terminalReward[1], 0, rs[i*2+1].reward, 0, numRfs);
                System.arraycopy(terminalDone[1], 0, rs[i*2+1].done, 0, numRfs);
                rs[i*2].done[0] = true;
                rs[i*2+1].done[0] = true;

                envSteps[i*2] =0;
                envSteps[i*2+1] =0;
            }
        }

        for (int i = selfPlayClients.length*2; i < players.length; i++) {
            envSteps[i] += 1;
            if (null == pool) {
                clients[i-selfPlayClients.length*2].gameStep(players[i]);
            }
            rs[i] = clients[i-selfPlayClients.length*2].getResponse();
            if (rs[i].done[0] || envSteps[i] >= maxSteps) {
                // stash previous values
                System.arraycopy(rs[i].reward, 0, terminalReward[0], 0, numRfs);
                System.arraycopy(rs[i].done, 0, terminalDone[0], 0, numRfs);

                clients[i-selfPlayClients.length*2].reset(players[i]);

                // restore previosly stashed values
                System.arraycopy(terminalReward[0], 0, rs[i].reward, 0, numRfs);
                System.arraycopy(terminalDone[0], 0, rs[i].done, 0, numRfs);
                rs[i].done[0] = true;

                envSteps[i] = 0;
            }
        }
        for (int i = 0; i < rs.length; i++) {
            reward[i] = rs[i].reward;
            done[i] = rs[i].done;
        }
        responses.set(null, reward, done);
        return responses;
    }

    public void getMasks(final int player) throws Exception {
        List<Future<Boolean>> selfPlayMaskResults = null;
        // if (pool != null) {
        //     final List<Callable<Boolean>> selfPlayMaskRequests = new ArrayList<>();
        //     for (int i = 0; i < selfPlayClients.length; i++) {
        //         final int clientIndex = i;
        //         selfPlayMaskRequests.add(() -> {
        //             selfPlayClients[clientIndex].getMasks(0);
        //             selfPlayClients[clientIndex].getMasks(1);
        //             return true;
        //         });
        //     }
        //     selfPlayMaskResults = pool.invokeAll(selfPlayMaskRequests);
        // }

        for (int i = 0; i < selfPlayClients.length; i++) {
            if (null == selfPlayMaskResults) {
                selfPlayClients[i].getMasks(0);
                selfPlayClients[i].getMasks(1);
            } else {
                selfPlayMaskResults.get(i).get();
            }
        }

        List<Future<Boolean>> maskResults = null;
        // if (pool != null) {
        //     final List<Callable<Boolean>> maskRequests = new ArrayList<>();
        //     for (int i = 0; i < clients.length; i++) {
        //         final int clientIndex = i;
        //         maskRequests.add(() -> {
        //             clients[clientIndex].getMasks(player);
        //             return true;
        //         });
        //     }
        //     maskResults = pool.invokeAll(maskRequests);
        // }

        for (int i = 0; i < clients.length; i++) {
            if (null == maskResults) {
                clients[i].getMasks(player);
            } else {
                maskResults.get(i).get();
            }
        }
    }

    public void close() throws Exception {
        if (clients != null) {
            for (JNIGridnetSharedMemClient client: clients) {
                client.close();
            }
        }
        if (selfPlayClients != null) {
            for (JNIGridnetSharedMemClientSelfPlay client: selfPlayClients) {
                client.close();
            }
        }

        // xxx(okachaiev): exchaange shutdown flag
        // if (pool != null) {
        //     pool.shutdownNow();
        // }
    }
}
