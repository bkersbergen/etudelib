package com.bol.etude.ng;

public abstract class Reporter<T,T2> {
    private final Persister<T2> persister;

    public Reporter(Persister<T2> persister) {
        this.persister = persister;
    }

    public final void report(T value) {
        persister.accept(transform(value));
    }

    protected abstract T2 transform(T value);
}
