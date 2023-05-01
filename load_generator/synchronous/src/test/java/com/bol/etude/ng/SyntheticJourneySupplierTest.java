package com.bol.etude.ng;

import com.bol.etude.dataproducer.SessionsLoader;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

class SyntheticJourneySupplierTest {

    @Test
    public void shouldDetermineParameters() throws IOException {
        String inputFile = "../../../etude/dataset/bolcom25m_sample/bolcom25m_sample.inter";
        List<Row> rows = SessionsLoader.readCsv(inputFile);
        SyntheticJourneySupplier sjs = new SyntheticJourneySupplier(20000000);
        sjs.fit(rows);
        for (int i = 0 ; i < 100; i++) {
            List<Long> clickedItems = sjs.get();
            System.out.println(clickedItems);
        }


    }

}
