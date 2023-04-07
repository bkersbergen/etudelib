package com.bol.etude.ng;

import javax.annotation.Nonnull;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.Supplier;

import static java.util.Collections.unmodifiableList;

class Journeys {

    private static long counter = 0;

    private final Supplier<List<Long>> supplier;

    Journeys(Supplier<List<Long>> supplier) {
        this.supplier = supplier;
    }
    private final ConcurrentLinkedQueue<Journey> journeys = new ConcurrentLinkedQueue<>();

    @Nonnull Journey pull() {
        Journey journey = journeys.poll();
        return journey == null ? new Journey(counter += 1, supplier.get()) : journey;
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
    }
}
