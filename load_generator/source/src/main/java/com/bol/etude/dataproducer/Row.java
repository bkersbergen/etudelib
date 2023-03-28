package com.bol.etude.dataproducer;

import java.util.Objects;

public class Row {

    public int sessionId;
    public long itemId;
    public int time;

    public Row(Integer sessionId, long itemId, int time) {
        this.sessionId = sessionId;
        this.itemId = itemId;
        this.time = time;
    }

    public int getSessionId() {
        return sessionId;
    }

    public long getItemId() {
        return itemId;
    }

    public int getTime(){
        return time;
    }

    public String toString() {
        return "SessionId: " + this.sessionId + " ItemId: " + this.itemId + " time: " + this.time;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Row row = (Row) o;
        return sessionId == row.sessionId &&
                itemId == row.itemId &&
                time == row.time;
    }

    @Override
    public int hashCode() {
        return Objects.hash(sessionId, itemId, time);
    }

}