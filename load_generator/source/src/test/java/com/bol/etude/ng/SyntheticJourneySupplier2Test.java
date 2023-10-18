package com.bol.etude.ng;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.checkerframework.checker.units.qual.A;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

class SyntheticJourneySupplier2Test {

    @Test
    void shouldFitPythonDistributionSessionLength() {
//        session_fit_params = (0.07004263982467046, 1.9999999999999998, 277.00000000000006)
//        item_fit_params = (0.040690006010909816, 0.9999999999999999, 2335.0000000000005)

        SyntheticJourneySupplier2 underTest = new SyntheticJourneySupplier2(10_000,
                0.07004263982467046, 1.9999999999999998, 277.00000000000006,
                0.040690006010909816, 0.9999999999999999, 2335.0000000000005
        );
        List<Integer> sessionsLengths = new ArrayList<>();
        for (int i = 0; i < 10_000; i++) {
            sessionsLengths.add(underTest.get().size());
        }
        Collections.sort(sessionsLengths);
        Percentile pMethod = new Percentile();
        double p1 = pMethod.evaluate(sessionsLengths.stream().mapToDouble(Integer::intValue).toArray(), 1);
        double p25 = pMethod.evaluate(sessionsLengths.stream().mapToDouble(Integer::intValue).toArray(), 25);
        double p50 = pMethod.evaluate(sessionsLengths.stream().mapToDouble(Integer::intValue).toArray(), 50);
        double p90 = pMethod.evaluate(sessionsLengths.stream().mapToDouble(Integer::intValue).toArray(), 90);
        double p99 = pMethod.evaluate(sessionsLengths.stream().mapToDouble(Integer::intValue).toArray(), 99);
        double p100 = pMethod.evaluate(sessionsLengths.stream().mapToDouble(Integer::intValue).toArray(), 100);
        System.out.println("real p1: " + p1 + " p25:" + p25 + " p50:" + p50 + " p90:" + p90 + " p99:" + p99);
        assertEquals(2.0, p1);
        assertEquals(2.0, p25);
        assertEquals(2.0, p50);
        assertEquals(62.0, p90);
        assertEquals(246.0, p99);
        assertEquals(277.0, p100);

    }


    @Test
    void shouldFitPythonDistributionItemFrequency() {
//        session_fit_params = (0.07004263982467046, 1.9999999999999998, 277.00000000000006)
//        item_fit_params = (0.040690006010909816, 0.9999999999999999, 2335.0000000000005)

        SyntheticJourneySupplier2 underTest = new SyntheticJourneySupplier2(10_000,
                0.07004263982467046, 1.9999999999999998, 277.00000000000006,
                0.040690006010909816, 0.9999999999999999, 2335.0000000000005
        );
        List<Long> clickedItems = new ArrayList<>();  /// only the item_ids. We ignore the sessions.
        for (int i = 0; i < 1_000; i++) {
            clickedItems.addAll(underTest.get());
        }

        Map<Long, Long> ff = clickedItems.stream()
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        System.out.println(ff.values());
    }
}