package com.bol.etude.torchserve;

import com.google.common.util.concurrent.RateLimiter;

import java.time.Duration;
import java.util.Timer;
import java.util.TimerTask;

public class RateLimiterFactory {

    private Timer rateLimiterTask;
    public RateLimiter createRateLimiter (double initialRate, int finalRequestedRate, Duration durationOfRampup) {
        final RateLimiter rateLimiter = RateLimiter.create(initialRate);  // Rate of http requests per second.
        TimerTask task = new DynamicRateLimiterAdjustmentTask(rateLimiter, finalRequestedRate, durationOfRampup);
        this.rateLimiterTask = new Timer("rateLimiterTask", false);
        rateLimiterTask.scheduleAtFixedRate(task, 0, 100);
        return rateLimiter;
    }

    public void stop() {
        this.rateLimiterTask.cancel();
    }

}
