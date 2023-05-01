package com.bol.etude.stoppingcondition;

public class RateLimitConfig {
    private final int startingValue;
    private final int targetValue;

    public RateLimitConfig(int startingValue, int targetValue) {
        this.startingValue = startingValue;
        this.targetValue = targetValue;
    }

    public int getStartingValue() {
        return startingValue;
    }

    public int getTargetValue() {
        return targetValue;
    }
}
