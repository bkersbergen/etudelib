package com.bol.etude.ng;

import com.bol.etude.ng.Requester.Response;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Collector<T> {

    private final HashMap<T, List<Response>> results = new HashMap<>();

    public void add(T key, Response value) {
        List<Response> values = results.computeIfAbsent(key, it -> new ArrayList<>());
        values.add(value);
    }

    public List<Response> remove(T key) {
        return results.remove(key);
    }
}
