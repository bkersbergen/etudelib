package com.bol.etude.dataproducer;

import me.tongfei.progressbar.ProgressBar;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

import static java.lang.Boolean.TRUE;

public class SytheticDataProducer implements SessiondataProducerTask {
    public BlockingQueue<UserSession> outputQueue;
    private Random rand = new Random();
    private volatile boolean keepRunning;
    private boolean isRealLoadTest;
    private static final Logger LOGGER = LogManager.getLogger(SytheticDataProducer.class);
    private volatile boolean runIsFinished;
    private long maxDurationInMs;

    public void configure(String training_csv_path, BlockingQueue<UserSession> outputQueue, boolean isRealLoadTest, Duration duration) {
        this.outputQueue = outputQueue;
        LOGGER.debug("started");
        this.keepRunning = true;
        this.runIsFinished = false;
        this.isRealLoadTest = isRealLoadTest;
        this.maxDurationInMs = duration.toMillis();
        boolean isEvaluationData = !isRealLoadTest;
        assert this.isRealLoadTest || this.maxDurationInMs > 0 : "invalid setup. Either isRealLoadTest needs to be true or set a max duration";
    }

    @Override
    public void run() {
        long initialMax = this.maxDurationInMs / 1000;
        String taskname = isRealLoadTest ? "Sessions" : "Ramping";
        try (ProgressBar pb = new ProgressBar(taskname, initialMax)) {
            long start = System.currentTimeMillis();
            UserSession userSession = createSythneticUserSession();
            while (keepRunning) {
                try {
                    boolean messageSend = outputQueue.offer(userSession, 100, TimeUnit.MILLISECONDS);
                    if (messageSend) {
                        if (System.currentTimeMillis() - start < this.maxDurationInMs) {
                            // we iterate over the dataset until this task ran for the max duration
                            pb.stepTo((System.currentTimeMillis() - start) / 1000);
                            userSession = createSythneticUserSession();
                        } else {
                            LOGGER.debug("Must stop. duration: {} < maxDurationInSeconds: {}", (System.currentTimeMillis() - start) / 1000, maxDurationInMs / 1000);
                            stop();
                        }
                    }
                    // message was send, so we prepare for the next session
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            pb.stepTo(initialMax);
        };

        this.runIsFinished = true;
    }

    private UserSession createSythneticUserSession() {
        List<Long> items = new ArrayList<>();
        List<Long> timestamps  = new ArrayList<>();
        for (int idx = 0 ; idx < rand.nextInt(10); idx++) {
            items.add((long) rand.nextInt(1000));
            timestamps.add((long) idx);
        }
        UserSession userSession = new UserSession(rand.nextLong(), items, timestamps, TRUE);
        return userSession;
    }

    @Override
    public void stop() {
        LOGGER.info("stopping data producer");
        keepRunning = false;
    }

    public boolean isRunIsFinished() {
        return runIsFinished;
    }
}
