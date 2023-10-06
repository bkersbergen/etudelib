package com.bol.etude.ng;

import com.bol.etude.generated.Meta;

import java.time.Duration;
import java.time.Instant;
import java.util.Iterator;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class Tester {

    private final Integer target;
    private final Duration ramp;
    private final long secondInMillis = Duration.ofSeconds(1).toMillis();
    private final Instant deadline;

    private Tester(Integer target, Duration ramp) {
        this.target = target;
        this.ramp = ramp;
        this.deadline = Instant.now().plus(ramp.plusMinutes(1));
    }

    private void run(Consumer<Request> runner, Persister<Meta> metaPersister) {
        long firstTickMillis = System.currentTimeMillis();
        long ticks = 0;
        AtomicInteger inflight = new AtomicInteger(0);
        Ramper ramper = new Ramper(target, ramp);

        outer:
        for (int rps : ramper) {
            ticks += 1;
            boolean first = true;
            long nextTickMillis = firstTickMillis + (ticks * secondInMillis);
            long millisToNextTick;

            if (isDeadlineReached()) {
                System.out.println("Tester.ticks['" + ticks + "'].deadline()");
                return;
            }

            System.out.println("Tester.ticks['" + ticks + "'].state(rps = '" + rps + "', inflight = '" + inflight.get() + "')");
            Meta t = Meta.newBuilder().setTimestampEpochMillis(Instant.now().toEpochMilli()).setTicks(ticks).setRps(rps).setInflight(inflight.get()).build();
            metaPersister.accept(t);
            for (int i = 0; i < rps; i++) {
                if (isDeadlineReached()) {
                    System.out.println("Tester.ticks['" + ticks + "'].deadline()");
                    return;
                }

                if (inflight.get() >= rps ){
                    System.out.println("Tester.ticks['" + ticks + "'].park(rps = '" + rps + "', inflight = '" + inflight.get() + "' iteration = '" + i + "')");
                    while (inflight.get() >= rps) {
                        if (millisTillNextTick(nextTickMillis) <= 0) {
                            continue outer;
                        }
                        Tester.sleep(1);
                    }
                }
                millisToNextTick = millisTillNextTick(nextTickMillis);
                if (millisToNextTick <= 0) {
                    System.out.println("Tester.ticks['" + ticks + "'].break(lag = '" + millisToNextTick + "', requestsInTick = '" + i + "', requestsToFillTick = '" + (rps - i) + "')");
                    continue outer;
                }

                runner.accept(new Request(ticks, rps, inflight, first));

                if (first) first = false;

                if (i + 1 < rps) {
                    millisToNextTick = millisTillNextTick(nextTickMillis);
                    if (millisToNextTick > 5) {
                        int requestsToFillTick = rps - i;
                        long msTillNextRequest = (long) (millisToNextTick / (double) requestsToFillTick);
                        if (msTillNextRequest > 2) {
                            Tester.sleep(msTillNextRequest);
                        }
                    }
                }
            }

            millisToNextTick = millisTillNextTick(nextTickMillis);

            if (millisToNextTick <= 0) {
                System.out.println("Tester.ticks['" + ticks + "'].lag('" + Duration.ofNanos(millisToNextTick).toSeconds() + "')");
                continue;
            }
            Tester.sleep(millisToNextTick);
        }
    }

    private static void sleep(long millis) {
        if (millis > 0) {
            try {
                Thread.sleep(millis);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private boolean isDeadlineReached() {
        return !Instant.now().isBefore(deadline);
    }

    private static long millisTillNextTick(long next) {
        return next - System.currentTimeMillis();
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

    public static void rampWithBackPressure(int target, Duration ramp, Persister<Meta> metaPersister, Consumer<Request> runner) throws InterruptedException {
        if (target < 0) throw new RuntimeException("target < 0");
        Objects.requireNonNull(ramp, "ramp == null");
        if (ramp.isNegative()) throw new IllegalArgumentException("ramp < 0");
        new Tester(target, ramp).run(runner, metaPersister);
    }
}
