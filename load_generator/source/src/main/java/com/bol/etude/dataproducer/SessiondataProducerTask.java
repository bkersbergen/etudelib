package com.bol.etude.dataproducer;

public interface SessiondataProducerTask extends Runnable {

    void run();

    void stop();
}
