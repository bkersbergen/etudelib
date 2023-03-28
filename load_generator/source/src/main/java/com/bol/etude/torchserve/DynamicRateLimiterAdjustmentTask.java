package com.bol.etude.torchserve;

import com.google.common.util.concurrent.RateLimiter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.Duration;
import java.util.TimerTask;

import static java.lang.Math.round;

public class DynamicRateLimiterAdjustmentTask extends TimerTask {
    /**
     * Adjust the rate of the ratelimiter in a linear way over the duration of time.
     */
    private static final Logger LOGGER = LogManager.getLogger(DynamicRateLimiterAdjustmentTask.class);
    private final RateLimiter rateLimiter;
    private final long startTimeMillis;
    private final int targetRate;
    private final long totalDurationMillis;

    public DynamicRateLimiterAdjustmentTask(RateLimiter rateLimiter, int targetRate, Duration durationOfRampup) {
        this.rateLimiter= rateLimiter;
        this.targetRate = targetRate;
        this.startTimeMillis = System.currentTimeMillis() - 1; // subtract one for fast machines that start the job in the same millisecond
        this.totalDurationMillis = durationOfRampup.toMillis();
    }

    @Override
    public void run() {
        long durationMillis = System.currentTimeMillis() - startTimeMillis;
        int permitsPerSecond = this.targetRate;
        if (durationMillis < totalDurationMillis) {
            double ratio = durationMillis / (double) totalDurationMillis;
            permitsPerSecond = (int)Math.round(permitsPerSecond * ratio);
            permitsPerSecond = Math.max(1, permitsPerSecond);
        }
        int actualRate = (int) Math.round(rateLimiter.getRate());
        if (actualRate != permitsPerSecond) {
            LOGGER.debug("rateLimiter.getRate() {} Adjusting rate limit to  {} ", actualRate, permitsPerSecond);
            rateLimiter.setRate(permitsPerSecond);
        }
    }
}
