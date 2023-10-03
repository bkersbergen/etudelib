package com.bol.etude.ng;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.time.Instant;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SyntheticJourneySupplierTest {

    @Test
    public void shouldDetermineSingleThreaded() {
        int C = 10_000_000;
        SyntheticJourneySupplier journeys = new SyntheticJourneySupplier(C);
        journeys.fit(5.597568416279968, 8.0E-5, 3.650557039874508);
        Journeys underTest = new Journeys(journeys);
        Instant start = Instant.now();
        int clicks = 0;
        for ( int i = 0 ; i < 10_000_000; i++) {
            Journeys.Journey journey = underTest.pull();
            for (int t = 0 ; t < journey.size(); t++) {
                List<Long> sessionItems = journey.item();
                clicks += 1;
            }
        }
        Duration duration = Duration.between(start, Instant.now());
        System.out.println(clicks / (duration.toSeconds()));
    }

}