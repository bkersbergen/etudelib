package com.bol.etude.ng;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

class CsvBasedJourneySupplierTest {
    @Test
    public void shouldReadFile() {
        CsvBasedJourneySupplier underTest = new CsvBasedJourneySupplier(1000000, "raw_click_lists.csv");
        for (int i = 0 ; i < 1000; i++) {
            List<Long> sessionItems = underTest.get();

            Assertions.assertTrue(sessionItems.size() > 0);
            System.out.println(sessionItems);
        }

    }


}