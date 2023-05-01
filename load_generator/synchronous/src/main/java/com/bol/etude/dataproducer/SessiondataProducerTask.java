package com.bol.etude.dataproducer;

import java.time.Duration;
import java.util.concurrent.BlockingQueue;

public interface SessiondataProducerTask extends Runnable {

    void run();

    void stop();

    void configure(String trainingDataPath, BlockingQueue<UserSession> sessionsQueue, boolean b, Duration duration);

    boolean isRunIsFinished();
}
