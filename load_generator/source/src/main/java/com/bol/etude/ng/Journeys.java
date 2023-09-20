package com.bol.etude.ng;

import org.checkerframework.checker.units.qual.A;

import javax.annotation.Nonnull;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;
import java.util.stream.LongStream;

import static java.util.Collections.unmodifiableList;

class Journeys {

    static Supplier<List<Long>> randomJourneySupplier() {
        return () -> LongStream.range(0, new Random().nextLong(1, 20)).boxed().toList();
    }

    private AtomicLong counter = new AtomicLong(0);
    private final Supplier<List<Long>> supplier;

    Journeys(Supplier<List<Long>> supplier) {
        this.supplier = supplier;
    }
    private final ConcurrentLinkedQueue<Journey> journeys = new ConcurrentLinkedQueue<>();

    @Nonnull Journey pull() {
        Journey journey = journeys.poll();

        if (journey != null) {
            return journey;
        } else {
            // expensive call
            List<Long> items = supplier.get();
            journey = new Journey(counter.getAndIncrement(), items);
            return journey;
        }
    }

    void push(@Nonnull Journey journey) {
        if (!journey.last()) {
            journeys.offer(journey);
        }
    }

    static class Journey {
        public final Long uid;

        private final List<Long> items;

        private int index = 0;


        public Journey(long uid, List<Long> items) {
            if (items.isEmpty()) throw new RuntimeException("items.isEmpty()");
            this.uid = uid;
            this.items = items;
        }

        @Nonnull public List<Long> item() {
            if (last()) throw new RuntimeException("index = " + index + ", size = " + items.size());
            index += 1;
            return items.subList(0, index);
        }

        @Nonnull public Boolean first() {
            return index == 0;
        }

        @Nonnull public Boolean last() {
            return index == items.size();
        }

        public int index() {
            return index;
        }

        public int size() {
            return items.size();
        }

        public List<Long> items() {
            return unmodifiableList(items);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Journey journey)) return false;
            return Objects.equals(uid, journey.uid) && Objects.equals(items, journey.items);
        }

        @Override
        public int hashCode() {
            return Objects.hash(uid, items);
        }

        @Override
        public String toString() {
            return "Journey{" +
                    "uid=" + uid +
                    ", items=" + items +
                    ", index=" + index +
                    '}';
        }
    }
}
