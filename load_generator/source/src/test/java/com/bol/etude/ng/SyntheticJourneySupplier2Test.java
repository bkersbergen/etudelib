package com.bol.etude.ng;

import org.checkerframework.checker.units.qual.A;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SyntheticJourneySupplier2Test {

    @Test
    void shouldFitPythonDistribution() {
//        session_fit_params = (0.07004263982467046, 1.9999999999999998, 277.00000000000006)
//        item_fit_params = (0.040690006010909816, 0.9999999999999999, 2335.0000000000005)

        SyntheticJourneySupplier2 underTest = new SyntheticJourneySupplier2(10_000,
                0.07004263982467046, 1.9999999999999998, 277.00000000000006,
                0.040690006010909816, 0.9999999999999999, 2335.0000000000005
                );
        List<Integer> sessionsLengths = new ArrayList<>();
        for (int i = 0 ; i < 10_000 ; i++ ) {
            sessionsLengths.add(underTest.get().size());
        }

    }
}