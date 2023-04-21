package com.bol.etude.ng;

import java.time.Duration;
import java.util.Iterator;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;

public class Tester {

    private final Integer target;
    private final Duration ramp;
    private final long milliInNanos = Duration.ofMillis(1).toNanos();
    private final long parkInNanos = milliInNanos / 10000; // 1 millis = 1_000_000 nanos
    private final long secondInNanos = Duration.ofSeconds(1).toNanos();

    private Tester(Integer target, Duration ramp) {
        this.target = target;
        this.ramp = ramp;
    }

    private void run(Consumer<Request> runner) throws InterruptedException {
        long firstTickMoment = System.nanoTime();
        long ticks = 0;
        AtomicInteger inflight = new AtomicInteger(0);
        Ramper ramper = new Ramper(target, ramp);

        outer:
        for (int rps : ramper) {
            ticks += 1;
            boolean first = true;
            long nextTickNanos = secondInNanos * ticks;
            long nextTickMoment = nextTickNanos + firstTickMoment;
            long timeToNextTick;

            System.out.println("Test.threads(active = '" + Thread.activeCount() + "'");

            for (int i = 0; i < rps; i++) {
                timeToNextTick = timeTillNextTick(nextTickMoment);

                while (inflight.get() == rps) {
                    if (timeToNextTick > milliInNanos) {
//                        System.out.println("Tester.ticks['" + ticks + "'].park(iteration = '" + i + "')");
                        LockSupport.parkNanos(parkInNanos);
                        timeToNextTick = timeTillNextTick(nextTickMoment);
                    }
                }

                if (timeToNextTick <= 0) {
                    long lag = Duration.ofNanos(timeToNextTick).toMillis();
                    System.out.println("Tester.ticks['" + ticks + "'].break(lag = '" + lag + "', ops = '" + i + "', noops = '" + (rps - i) + "')");
                    continue outer;
                }

//                long scheduleTaskStart = System.nanoTime();
                runner.accept(new Request(ticks, rps, inflight, first));
//                long scheduleTaskNanos = System.nanoTime() - scheduleTaskStart;

                if (first) first = false;
            }

            timeToNextTick = timeTillNextTick(nextTickMoment);

            if (timeToNextTick <= 0) {
                System.out.println("Tester.ticks['" + ticks + "'].lag('" + Duration.ofNanos(timeToNextTick).toSeconds() + "')");
                continue;
            }

            long millis = Duration.ofNanos(timeToNextTick).toMillis();
            Thread.sleep(millis);
        }
    }

    private static long timeTillNextTick(long next) {
        return next - System.nanoTime();
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
        private boolean flying = false;

        Request(long ticks, long rps, AtomicInteger inflight, boolean start) {
            this.ticks = ticks;
            this.rps = rps;
            this.inflight = inflight;
            this.start = start;
        }

        public void doOnTickStart(Runnable runnable) {
            if (ticks % 10 == 0 && start) {
                try {
                    runnable.run();
                } catch (Throwable t) {
                    System.out.println("Ticker.doOnTickStart().err['" + t.getClass().getSimpleName() + "']");
                    t.printStackTrace();
                }
            }
        }

        public void fly() {
            if (!flying) {
                inflight.incrementAndGet();
                flying = true;
            }
        }

        public void unfly() {
            if (flying) {
                inflight.decrementAndGet();
                flying = false;
            }
        }
    }

    public static void rampWithBackPressure(int target, Duration ramp, Consumer<Request> runner) throws InterruptedException {
        if (target < 0) throw new RuntimeException("target < 0");
        Objects.requireNonNull(ramp, "ramp == null");
        if (ramp.isNegative()) throw new IllegalArgumentException("ramp < 0");
        new Tester(target, ramp).run(runner);
    }
}
