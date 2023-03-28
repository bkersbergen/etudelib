package com.bol.etude.dataproducer;

import java.util.List;

public class UserSession {
    private long sessionId;
    private List<Long> items;
    private List<Long> itemTimestamps;
    private boolean isEvaluationData;

    public UserSession(long sessionId, List<Long> items, List<Long> itemTimestamps, boolean isEvaluationData) {
        this.sessionId = sessionId;
        this.items = items;
        this.itemTimestamps = itemTimestamps;
        this.isEvaluationData = isEvaluationData;
    }

    public long getSessionId() {
        return sessionId;
    }

    public List<Long> getItems() {
        return items;
    }

    public List<Long> getItemTimestamps() {
        return itemTimestamps;
    }

    public boolean isEvaluationData() {
        return isEvaluationData;
    }
}
