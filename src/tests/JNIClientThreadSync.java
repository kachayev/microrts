package tests;

import java.util.concurrent.CountDownLatch;

/**
 * Utility class to carry synchronization locks to run vec env
 * clients on separate threads
 */
public class JNIClientThreadSync {

    public static enum OpType {
        STEP,
        MASKS,
        SHUTDOWN,
    }

    final CountDownLatch blocker;
    final CountDownLatch ready;
    final OpType op;

    private JNIClientThreadSync(CountDownLatch blocker, CountDownLatch ready, OpType op) {
        this.blocker = blocker;
        this.ready = ready;
        this.op = op;
    }

    public final static JNIClientThreadSync forClients(int numClients, OpType op) {
        return new JNIClientThreadSync(new CountDownLatch(1), new CountDownLatch(numClients), op);
    }

    public CountDownLatch getBlockerLock() {
        return this.blocker;
    }

    public CountDownLatch getReadyLock() {
        return this.ready;
    }

    public OpType getOpType() {
        return this.op;
    }

}