package com.bol.etude.replayer;

public interface ResultWriterTask extends Runnable {
    void run();

    void stop();
}
