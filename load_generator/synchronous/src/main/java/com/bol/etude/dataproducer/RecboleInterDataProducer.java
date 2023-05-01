package com.bol.etude.dataproducer;

import me.tongfei.progressbar.ProgressBar;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.Duration;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

public class RecboleInterDataProducer implements SessiondataProducerTask {
    public BlockingQueue<UserSession> outputQueue;
    private List<UserSession> userSessions;
    private volatile boolean keepRunning;
    private boolean isRealLoadTest;
    private static final Logger LOGGER = LogManager.getLogger(RecboleInterDataProducer.class);
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
        userSessions = SessionsLoader.getUserSessions(training_csv_path, isEvaluationData);
    }

    @Override
    public void run() {
        Iterator<UserSession> sessionIterator = userSessions.iterator();
        long initialMax = isRealLoadTest ? userSessions.size() : (this.maxDurationInMs / 1000);
        String taskname = isRealLoadTest ? "Sessions" : "Ramping";
        try (ProgressBar pb = new ProgressBar(taskname, initialMax)) {
            UserSession userSession = sessionIterator.next();
            long start = System.currentTimeMillis();
            while (keepRunning) {
                try {
                    boolean messageSend = outputQueue.offer(userSession, 100, TimeUnit.MILLISECONDS);
                    if (messageSend) {
                        if (isRealLoadTest) {
                            // we iterate over the dataset until the iterator is done
                            if (sessionIterator.hasNext()) {
                                pb.step();
                                userSession = sessionIterator.next();
                            } else {
                                LOGGER.debug("Must stop. Iterator is done");
                                stop();
                            }
                        } else if (System.currentTimeMillis() - start < this.maxDurationInMs) {
                            // we iterate over the dataset until this task ran for the max duration
                            pb.stepTo((System.currentTimeMillis() - start) / 1000);
                            if (!sessionIterator.hasNext()) {
                                sessionIterator = userSessions.iterator();
                            }
                            userSession = sessionIterator.next();
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


    @Override
    public void stop() {
        LOGGER.info("stopping data producer");
        keepRunning = false;
    }

    public boolean isRunIsFinished() {
        return runIsFinished;
    }
}
