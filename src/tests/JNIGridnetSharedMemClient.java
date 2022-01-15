/*
* To change this template, choose Tools | Templates
* and open the template in the editor.
*/
package tests;

import java.io.Writer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.awt.image.BufferedImage;
import java.io.StringWriter;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import com.beust.jcommander.Parameter;

import ai.PassiveAI;
import ai.RandomBiasedAI;
import ai.RandomNoAttackAI;
import ai.core.AI;
import ai.jni.JNIAI;
import ai.rewardfunction.RewardFunctionInterface;
import ai.jni.JNIInterface;
import ai.jni.Response;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import rts.GameState;
import rts.PartiallyObservableGameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.Trace;
import rts.TraceEntry;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitTypeTable;
import util.NDBuffer;
import weka.core.pmml.jaxbbindings.False;

/**
 *
 * Improved performance for JVM <-> NumPy data exchange
 * with direct buffer (JVM allocated).
 * 
 */
public class JNIGridnetSharedMemClient implements Runnable {

    // Settings
    public RewardFunctionInterface[] rfs;
    String micrortsPath;
    String mapPath;
    public AI ai2;
    UnitTypeTable utt;
    public boolean partialObs = false;

    // Internal State
    PhysicalGameState pgs;
    GameState gs;
    GameState player1gs, player2gs;
    boolean gameover = false;
    boolean layerJSON = true;
    public int renderTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;
    public int maxAttackRadius;
    PhysicalGameStateJFrame w;
    public JNIAI ai1;

    final int clientOffset;

    // storage
    final NDBuffer obsBuffer;
    final NDBuffer actionMaskBuffer;
    final NDBuffer actionBuffer;
    double[] rewards;
    boolean[] dones;
    Response response;
    PlayerAction pa1;
    PlayerAction pa2;

    // sync
    final CountDownLatch clientReadyCounter;
    final AtomicReference<JNIClientThreadSync> clientSyncRef;

    public JNIGridnetSharedMemClient(RewardFunctionInterface[] a_rfs, String mapPath, AI a_ai2, UnitTypeTable a_utt, boolean partial_obs,
            int clientOffset, NDBuffer obsBuffer, NDBuffer actionMaskBuffer, NDBuffer actionBuffer,
            CountDownLatch clientReadyCounter, AtomicReference<JNIClientThreadSync> clientSyncRef) throws Exception{
        this.clientOffset = clientOffset;
        this.obsBuffer = obsBuffer;
        this.actionMaskBuffer = actionMaskBuffer;
        this.actionBuffer = actionBuffer;
        this.clientReadyCounter = clientReadyCounter;
        this.clientSyncRef = clientSyncRef;
        this.mapPath = mapPath;
        partialObs = partial_obs;

        rfs = a_rfs;
        utt = a_utt;
        pgs = PhysicalGameState.load(mapPath, utt);
        ai1 = new JNIAI(100, 0, utt);
        ai2 = a_ai2;
        if (ai2 == null) {
            throw new Exception("no ai2 was chosen");
        }

        // initialize storage
        rewards = new double[rfs.length];
        dones = new boolean[rfs.length];
        response = new Response(null, null, null, null);
    }

    public byte[] render(boolean returnPixels) throws Exception {
        if (w==null) {
            w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, partialObs, null, renderTheme);
        }
        w.setStateCloning(gs);
        w.repaint();

        if (!returnPixels) {
            return null;
        }
        BufferedImage image = new BufferedImage(w.getWidth(),
            w.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
            w.paint(image.getGraphics());

        WritableRaster raster = image.getRaster();
        DataBufferByte data = (DataBufferByte) raster.getDataBuffer();
        return data.getData();
    }

    public void gameStep(int player) throws Exception {
        if (partialObs) {
            player1gs = new PartiallyObservableGameState(gs, player);
            player2gs = new PartiallyObservableGameState(gs, 1 - player);
        } else {
            player1gs = gs;
            player2gs = gs;
        }
        pa1 = ai1.getActionFromBuffer(player, player1gs, clientOffset, actionBuffer);
        pa2 = ai2.getAction(1 - player, player2gs);
        gs.issueSafe(pa1);
        gs.issueSafe(pa2);
        TraceEntry te  = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
        te.addPlayerAction(pa1.clone());
        te.addPlayerAction(pa2.clone());

        // simulate:
        gameover = gs.cycle();
        if (gameover) {
            // ai1.gameOver(gs.winner());
            ai2.gameOver(gs.winner());
        }
        for (int i = 0; i < rewards.length; i++) {
            rfs[i].computeReward(player, 1 - player, te, gs);
            dones[i] = rfs[i].isDone();
            rewards[i] = rfs[i].getReward();
        }

        player1gs.getBufferObservation(player, clientOffset, obsBuffer);
        response.set(null, rewards, dones, ai1.computeInfo(player, player2gs));
    }

    public void getMasks(int player) throws Exception {
        actionMaskBuffer.resetSegment(new int[]{clientOffset});

        for (int i = 0; i < pgs.getUnits().size(); i++) {
            Unit u = pgs.getUnits().get(i);
            UnitActionAssignment uaa = gs.getUnitActions().get(u);
            if (u.getPlayer() == player && uaa == null) {
                final int[] idxOffset = new int[]{clientOffset, u.getY(), u.getX()};
                UnitAction.getValidActionBuffer(u, gs, utt, actionMaskBuffer, maxAttackRadius, idxOffset);
            }
        }
    }

    public String sendUTT() throws Exception {
        Writer w = new StringWriter();
        utt.toJSON(w);
        return w.toString(); // now it works fine
    }

    public void reset(int player) throws Exception {
        ai1.reset();
        ai2 = ai2.clone();
        ai2.reset();
        pgs = PhysicalGameState.load(mapPath, utt);
        gs = new GameState(pgs, utt);
        if (partialObs) {
            player1gs = new PartiallyObservableGameState(gs, player);
        } else {
            player1gs = gs;
        }

        for (int i = 0; i < rewards.length; i++) {
            rewards[i] = 0;
            dones[i] = false;
        }

        player1gs.getBufferObservation(player, clientOffset, obsBuffer);
        response.set(null, rewards, dones, "{}");
    }

    public Response getResponse() {
        return response;
    }

    public void close() throws Exception {
        if (w != null) {
            w.dispose();    
        }
    }

    @Override
    public void run() {
        JNIClientThreadSync clientSync = null;
        clientReadyCounter.countDown();
        boolean running = true;
        while (running) {
            try {
                clientSync = clientSyncRef.get();
                clientSync.getBlockerLock().await();
                switch (clientSync.getOpType()) {
                    case STEP:
                        gameStep(0);
                        break;

                    case MASKS:
                        // xxx(okachaiev): seems like this is always 0, right?
                        getMasks(0);
                        break;

                    case SHUTDOWN:
                        running = false;
                        break;
                }
            } catch(InterruptedException e) {
                // no-op
                // xxx(okachaiev): very wrong!
                running = false;
            } catch(Exception e) {
                e.printStackTrace();
                // xxx(okachaiev): again no op?
            } finally {
                if (null != clientSync) {
                    clientSync.getReadyLock().countDown();
                }
            }
        }
    }
}
