package com.bol.etude.ng;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class BolcomJourneySupplierTest {
    @Test
    public void shouldReadFile() {
        BolcomJourneySupplier underTest = new BolcomJourneySupplier(500);
        List<Long> sessionItems = underTest.get();
        Assertions.assertTrue(sessionItems.size() > 0);
    }

}