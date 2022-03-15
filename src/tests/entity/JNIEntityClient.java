/*
* To change this template, choose Tools | Templates
* and open the template in the editor.
*/
package tests.entity;

import java.io.Writer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.awt.image.BufferedImage;
import java.io.StringWriter;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import com.beust.jcommander.Parameter;

import ai.PassiveAI;
import ai.RandomBiasedAI;
import ai.RandomNoAttackAI;
import ai.core.AI;
import ai.jni.EntityResponse;
import ai.jni.JNIAI;
import ai.rewardfunction.RewardFunctionInterface;
import ai.jni.JNIInterface;
import ai.jni.EntityResponse;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import rts.GameState;
import rts.PartiallyObservableGameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.ResourceUsage;
import rts.Trace;
import rts.TraceEntry;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitTypeTable;
import weka.core.pmml.jaxbbindings.False;

/**
 *
 * @author santi
 * 
 *         Once you have the server running (for example, run
 *         "RunServerExample.java"), set the proper IP and port in the variable
 *         below, and run this file. One of the AIs (ai1) is run remotely using
 *         the server.
 * 
 *         Notice that as many AIs as needed can connect to the same server. For
 *         example, uncomment line 44 below and comment 45, to see two AIs using
 *         the same server.
 * 
 */
public class JNIEntityClient {

    // Settings
    public RewardFunctionInterface[] rfs;
    String micrortsPath;
    public String mapPath;
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
    public JNIInterface ai1;

    // storage
    int[][][] masks;
    double[] rewards;
    boolean[] dones;
    EntityResponse response;
    PlayerAction pa1;
    PlayerAction pa2;
    int centerCoordinate;

    public JNIEntityClient(RewardFunctionInterface[] a_rfs, String a_micrortsPath, String a_mapPath, AI a_ai2, UnitTypeTable a_utt, boolean partial_obs) throws Exception{
        micrortsPath = a_micrortsPath;
        mapPath = a_mapPath;
        rfs = a_rfs;
        utt = a_utt;
        partialObs = partial_obs;
        maxAttackRadius = utt.getMaxAttackRange() * 2 + 1;
        centerCoordinate = maxAttackRadius / 2;
        ai1 = new JNIAI(100, 0, utt);
        ai2 = a_ai2;
        if (ai2 == null) {
            throw new Exception("no ai2 was chosen");
        }
        if (micrortsPath.length() != 0) {
            this.mapPath = Paths.get(micrortsPath, mapPath).toString();
        }

        pgs = PhysicalGameState.load(mapPath, utt);

        // initialize storage
        masks = new int[pgs.getHeight()][pgs.getWidth()][1+6+4+4+4+4+utt.getUnitTypes().size()+maxAttackRadius*maxAttackRadius];
        rewards = new double[rfs.length];
        dones = new boolean[rfs.length];
        response = new EntityResponse(null, null, null, null);
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

        WritableRaster raster = image .getRaster();
        DataBufferByte data = (DataBufferByte) raster.getDataBuffer();
        return data.getData();
    }

    public EntityResponse gameStep(
        int[] unitActionActors,
        int[] unitActions,
        int[] baseActionActors,
        int[] baseActions,
        int[] barrackActionActors,
        int[] barrackActions,
        int player
    ) throws Exception {
        if (partialObs) {
            player1gs = new PartiallyObservableGameState(gs, player);
            player2gs = new PartiallyObservableGameState(gs, 1 - player);
        } else {
            player1gs = gs;
            player2gs = gs;
        }
        // pa1 = ai1.getAction(player, player1gs, unitActions);
        PlayerAction pa1 = new PlayerAction();
        ResourceUsage base_ru = new ResourceUsage();
		for (Unit u : gs.getPhysicalGameState().getUnits()) {
			UnitActionAssignment uaa = gs.unitActions.get(u);
			if (uaa != null) {
				ResourceUsage ru = uaa.action.resourceUsage(u, gs.getPhysicalGameState());
				base_ru.merge(ru);
			}
        }
        pa1.setResourceUsage(base_ru.clone());

        for (int i = 0; i < unitActionActors.length; i++) {
            Unit u = gs.pgs.getUnit(unitActionActors[i]);
            UnitAction ua;
            if (unitActions[i] < 4) {
                ua = new UnitAction(UnitAction.TYPE_MOVE, unitActions[i]);
            } else if (unitActions[i] >= 4 && unitActions[i] < 8) {
                ua = new UnitAction(UnitAction.TYPE_HARVEST, unitActions[i] - 4);
            } else if (unitActions[i] >= 8 && unitActions[i] < 12) {
                ua = new UnitAction(UnitAction.TYPE_RETURN, unitActions[i] - 8);
            } else if (unitActions[i] >= 12 && unitActions[i] < 16) {
                ua = new UnitAction(UnitAction.TYPE_PRODUCE, unitActions[i] - 12, utt.getUnitType("Base"));
            } else if (unitActions[i] >= 16 && unitActions[i] < 20) {
                ua = new UnitAction(UnitAction.TYPE_PRODUCE, unitActions[i] - 16, utt.getUnitType("Barracks"));
            } else {
                int relative_x = ((unitActions[i] - 20) % maxAttackRadius - centerCoordinate);
                int relative_y = ((unitActions[i] - 20) / maxAttackRadius - centerCoordinate);
                // System.out.println("u.getX()" + u.getX() + " u.getY(): " + u.getY());
                // System.out.println("relative_x: " + relative_x + " relative_y: " + relative_y);
                ua = new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, u.getX() + relative_x, u.getY() + relative_y);
                // System.out.println(ua);
            }
            // System.out.println("u");
            // System.out.println(u);
            // System.out.println("ua");
            // System.out.println(ua);
            if (ua.resourceUsage(u, gs.pgs).consistentWith(pa1.getResourceUsage(), gs)) {
                ResourceUsage ru = ua.resourceUsage(u, gs.pgs);
                pa1.getResourceUsage().merge(ru);                        
                pa1.addUnitAction(u, ua);
            }
        }
        for (int i = 0; i < baseActionActors.length; i++) {
            Unit u = gs.pgs.getUnit(baseActionActors[i]);
            UnitAction ua;
            ua = new UnitAction(UnitAction.TYPE_PRODUCE, baseActions[i], utt.getUnitType("Worker"));
            // System.out.println("u");
            // System.out.println(u);
            // System.out.println("ua");
            // System.out.println(ua);
            if (ua.resourceUsage(u, gs.pgs).consistentWith(pa1.getResourceUsage(), gs)) {
                ResourceUsage ru = ua.resourceUsage(u, gs.pgs);
                pa1.getResourceUsage().merge(ru);                        
                pa1.addUnitAction(u, ua);
            }
        }
        for (int i = 0; i < barrackActionActors.length; i++) {
            Unit u = gs.pgs.getUnit(barrackActionActors[i]);
            UnitAction ua;
            if (barrackActions[i] < 4) {
                ua = new UnitAction(UnitAction.TYPE_PRODUCE, barrackActions[i], utt.getUnitType("Light"));
            } else if (barrackActions[i] >= 4 && barrackActions[i] < 8) {
                ua = new UnitAction(UnitAction.TYPE_PRODUCE, barrackActions[i] - 4, utt.getUnitType("Heavy"));
            } else {
                ua = new UnitAction(UnitAction.TYPE_PRODUCE, barrackActions[i] - 8, utt.getUnitType("Ranged"));
            }
            // System.out.println("u");
            // System.out.println(u);
            // System.out.println("ua");
            // System.out.println(ua);
            if (ua.resourceUsage(u, gs.pgs).consistentWith(pa1.getResourceUsage(), gs)) {
                ResourceUsage ru = ua.resourceUsage(u, gs.pgs);
                pa1.getResourceUsage().merge(ru);                        
                pa1.addUnitAction(u, ua);
            }
        }
        pa1.fillWithNones(gs, player, 1);
        // System.out.println("pa1");
        // System.out.println(pa1);
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
        response.set(
            ai1.getEntityObservation(player, player1gs),
            rewards,
            dones,
            ai1.computeInfo(player, player2gs));
        return response;
    }

    public String sendUTT() throws Exception {
        Writer w = new StringWriter();
        utt.toJSON(w);
        return w.toString(); // now it works fine
    }

    public EntityResponse reset(int player) throws Exception {
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
        response.set(
            ai1.getEntityObservation(player, player1gs),
            rewards,
            dones,
            "{}");
        return response;
    }

    public void close() throws Exception {
        if (w!=null) {
            w.dispose();    
        }
    }
}
