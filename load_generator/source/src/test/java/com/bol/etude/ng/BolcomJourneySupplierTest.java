package com.bol.etude.ng;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class BolcomJourneySupplierTest {
    @Test
    public void shouldReadFile() {
        BolcomJourneySupplier underTest = new BolcomJourneySupplier(1000000);
        for (int i = 0 ; i < 1000; i++) {
            List<Long> sessionItems = underTest.get();

            Assertions.assertTrue(sessionItems.size() > 0);
            System.out.println(sessionItems);
        }

    }


}