package com.bol.etude.ng;

import com.bol.etude.ng.Requester.Response;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

public class Collector<T> {

    private final ConcurrentHashMap<T, List<Response>> results = new ConcurrentHashMap<>();

    public void add(T key, Response value) {
        List<Response> values = results.computeIfAbsent(key, it -> new ArrayList<>());
        values.add(value);
    }

    public List<Response> remove(T key) {
        return results.remove(key);
    }
}
