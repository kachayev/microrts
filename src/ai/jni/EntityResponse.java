package ai.jni;

import java.util.ArrayList;

public class EntityResponse {

    public ArrayList[] observation;
    public double[] reward;
    public boolean[] done;
    public String info;

    public EntityResponse(ArrayList[] observation, double reward[], boolean done[], String info) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = info;
    }

    public void set(ArrayList[] observation, double reward[], boolean done[], String info) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = info;
    }
}