package com.bol.etude.ng;

import java.io.Closeable;
import java.io.Flushable;
import java.util.function.Consumer;

public interface Persister<T> extends Consumer<T>, Closeable, Flushable {}
