package com.bol.etude.ng;

import java.time.Duration;
import java.util.Iterator;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class Tester implements Iterable<Integer> {

    private final Integer target;
    private final Duration ramp;
    private final Duration maintain;

    private Tester(Integer target, Duration ramp, Duration maintain) {
        this.target = target;
        this.ramp = ramp;
        this.maintain = maintain;
    }

    @Override
    public Iterator<Integer> iterator() {
        Ramper ramper = new Ramper(target, ramp);
        Maintainer maintainer = new Maintainer(target, maintain);
        return new Iterators(ramper, maintainer);
    }

    public static void rampThenHold(int target, Duration ramp, Duration maintain, Consumer<Request> runner) throws InterruptedException {
        if (target < 0) throw new RuntimeException("target < 0");
        Objects.requireNonNull(ramp, "ramp == null");
        Objects.requireNonNull(ramp, "maintain == null");
        new Tester(target, ramp, maintain).run(runner);
    }

    private void run(Consumer<Request> runner) throws InterruptedException {
        long start = System.nanoTime();
        long ticks = 0;
        AtomicInteger inflight = new AtomicInteger(0);

        for (int rps : this) {
            ticks += 1;
            boolean first = true;

            for (int i = 0; i < rps; i++) {
                if (inflight.get() == rps) {
//                    System.out.println("Ticker.skip(tick = '" + ticks + "', num = '" + i + "')");
                    continue;
                }

                runner.accept(new Request(ticks, rps, inflight, first));
                if (first) first = false;
            }

            long next = Duration.ofSeconds(1).toNanos() * ticks;
            long delta = (start + next) - System.nanoTime();

            if (delta < 0) {
                System.out.println("Ticker.delta(tick = '" + ticks + "', 'seconds = " + Duration.ofNanos(delta).toSeconds() + ")");
                continue;
            }

            long millis = Duration.ofNanos(delta).toMillis();
            Thread.sleep(millis);
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
                System.out.println("Ticker.onTickStart(tick = '" + ticks + "', rps = '" + rps + "', inflight = '" + inflight.get() + "')");
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

    private static class Ramper implements Iterator<Integer> {
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
    }

    private static class Maintainer implements Iterator<Integer> {

        private final int value;
        private final Duration time;
        private long start = 0;

        Maintainer(int value, Duration time) {
            this.value = value;
            this.time = time;
        }

        @Override
        public boolean hasNext() {
            if (start == 0) start = System.nanoTime();
            return System.nanoTime() < start + time.toNanos();
        }

        @Override
        public Integer next() {
            return value;
        }
    }

    private static class Iterators implements Iterator<Integer> {
        private final Iterator<Integer> first;
        private final Iterator<Integer> last;

        Iterators(Iterator<Integer> first, Iterator<Integer> last) {
            this.first = first;
            this.last = last;
        }

        @Override
        public boolean hasNext() {
            return first.hasNext() || last.hasNext();
        }

        @Override
        public Integer next() {
            return first.hasNext() ? first.next() : last.next();
        }
    }
}
