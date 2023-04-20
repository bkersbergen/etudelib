package com.bol.etude.ng;

import java.time.Duration;
import java.util.Iterator;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class Tester {

    private final Integer target;
    private final Duration ramp;

    private Tester(Integer target, Duration ramp) {
        this.target = target;
        this.ramp = ramp;
    }

    private void run(Consumer<Request> runner) throws InterruptedException {
        long start = System.nanoTime();
        long ticks = 0;
        AtomicInteger inflight = new AtomicInteger(0);
        Ramper ramper = new Ramper(target, ramp);

        for (int rps : ramper) {
            ticks += 1;
            boolean first = true;

            for (int i = 0; i < rps; i++) {
                if (inflight.get() == rps) {
                    continue;
                }

                runner.accept(new Request(ticks, rps, inflight, first));
                if (first) first = false;
            }

            long next = Duration.ofSeconds(1).toNanos() * ticks;
            long delta = (start + next) - System.nanoTime();

            if (delta < 0) {
                continue;
            }

            long millis = Duration.ofNanos(delta).toMillis();
            Thread.sleep(millis);
        }
    }

    private static class Ramper implements Iterable<Integer>, Iterator<Integer> {
        private final float target;
        private final float size;

        Ramper(float target, Duration time) {
            this.target = target;
            this.size = target / time.toNanos();
        }

        private long start = 0;
        private float current = 0;

        @Override
        public boolean hasNext() {
            if (start == 0) start = System.nanoTime();
            return current != target;
        }

        @Override
        public Integer next() {
            current = Math.min(target, (System.nanoTime() - start) * size);
            return (int) Math.ceil(current);
        }

        @Override
        public Iterator<Integer> iterator() {
            return this;
        }
    }

    static class Request {
        private final long ticks;
        private final long rps;
        private final AtomicInteger inflight;
        private final boolean start;

        Request(long ticks, long rps, AtomicInteger inflight, boolean start) {
            this.ticks = ticks;
            this.rps = rps;
            this.inflight = inflight;
            this.start = start;
        }

        public void doOnTickStart(Runnable runnable) {
            if (ticks % 10 == 0 && start) {
                runnable.run();
            }
        }

        public void start() {
            inflight.incrementAndGet();
        }

        public void complete() {
            inflight.decrementAndGet();
        }
    }

    public static void rampWithBackPressure(int target, Duration ramp, Consumer<Request> runner) throws InterruptedException {
        if (target < 0) throw new RuntimeException("target < 0");
        Objects.requireNonNull(ramp, "ramp == null");
        if (ramp.isNegative()) throw new IllegalArgumentException("ramp < 0");
        new Tester(target, ramp).run(runner);
    }
}
