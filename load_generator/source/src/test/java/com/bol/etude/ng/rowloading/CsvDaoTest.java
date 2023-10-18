//package com.bol.etude.ng.rowloading;
//
//import com.bol.etude.ng.SyntheticJourneySupplier;
//import org.apache.logging.log4j.core.util.Assert;
//import org.junit.jupiter.api.Test;
//
//import java.util.List;
//
//import static org.junit.jupiter.api.Assertions.*;
//
//class CsvDaoTest {
//
//    @Test
//    public void shouldReadLines() {
//        CsvDao csv = new CsvDao("yoochoose_sample.csv");
//        List<SyntheticJourneySupplier.Row> rows = csv.readRows();
//        Assert.valueIsAtLeast(rows.size(), 1000000);
//    }
//
//}