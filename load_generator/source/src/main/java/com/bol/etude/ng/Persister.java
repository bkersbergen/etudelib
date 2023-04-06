package com.bol.etude.ng;

import java.io.Closeable;
import java.util.function.Consumer;

public interface Persister<T> extends Consumer<T>, Closeable {}
