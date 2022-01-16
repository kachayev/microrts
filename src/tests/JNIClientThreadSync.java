package tests;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Utility class to carry synchronization locks to run vec env
 * clients on separate threads
 */
public class JNIClientThreadSync {

    public static enum OpType {
        NOOP,
        STEP,
        MASKS,
        SHUTDOWN,
    }

    final CountDownLatch blocker;
    final CountDownLatch ready;
    final AtomicReference<OpType> nextOp;

    private JNIClientThreadSync(CountDownLatch blocker, CountDownLatch ready, AtomicReference<OpType> nextOp) {
        this.blocker = blocker;
        this.ready = ready;
        this.nextOp = nextOp;
    }

    public final static JNIClientThreadSync forClients(int numClients) {
        final AtomicReference<OpType> nextOp = new AtomicReference<>(OpType.NOOP);
        return new JNIClientThreadSync(new CountDownLatch(1), new CountDownLatch(numClients), nextOp);
    }

    public final static void executeOn(AtomicReference<JNIClientThreadSync> ref, OpType op, int numClients) {
        // replace sync object for the next cycle
        final JNIClientThreadSync currentSync = ref.getAndSet(forClients(numClients));

        currentSync.setNextOpType(op);
        // let all threads run
        currentSync.getBlockerLock().countDown();
        try {
            // wait for all threads to finish
            currentSync.getReadyLock().await();
        } catch (InterruptedException e) {
            throw new RuntimeException("Sync thread was interrupted.");
        }
    }

    public CountDownLatch getBlockerLock() {
        return this.blocker;
    }

    public CountDownLatch getReadyLock() {
        return this.ready;
    }

    public OpType getNextOpType() {
        return this.nextOp.get();
    }

    public void setNextOpType(OpType op) {
        this.nextOp.set(op);
    }

}