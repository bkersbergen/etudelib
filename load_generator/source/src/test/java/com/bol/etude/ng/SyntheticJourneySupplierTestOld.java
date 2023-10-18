//package com.bol.etude.ng;
//
//import com.bol.etude.ng.rowloading.CsvDao;
//import org.apache.commons.math3.random.RandomDataGenerator;
//import org.apache.commons.math3.random.Well19937c;
//import org.checkerframework.checker.units.qual.A;
//import org.junit.jupiter.api.Assertions;
//import org.junit.jupiter.api.Test;
//import org.apache.commons.math3.distribution.ZipfDistribution;
//import org.apache.commons.math3.distribution.IntegerDistribution;
//
//import org.apache.commons.math3.random.RandomGenerator;
//import org.apache.commons.math3.random.RandomGeneratorFactory;
//import java.util.Random;
//
//import org.apache.commons.math3.stat.descriptive.rank.Percentile;
//
//import java.time.Duration;
//import java.time.Instant;
//import java.util.*;
//import java.util.function.Function;
//import java.util.stream.Collectors;
//
//import org.apache.commons.math3.distribution.*;
//import org.apache.commons.math3.fitting.*;
//import org.apache.commons.math3.stat.*;
//
//
//class SyntheticJourneySupplierTest {
//
//    @Test
//    public void shouldDetermineSingleThreaded() {
//        int C = 10_000;
//        SyntheticJourneySupplier journeys = new SyntheticJourneySupplier(C);
////        journeys.fit(5.597568416279968, 8.0E-5, 3.650557039874508);
//        Journeys underTest = new Journeys(journeys);
//        Instant start = Instant.now();
//        int clicks = 0;
//        for (int i = 0; i < 1_000; i++) {
//            Journeys.Journey journey = underTest.pull();
//            for (int t = 0; t < journey.size(); t++) {
//                List<Long> sessionItems = journey.item();
//                clicks += 1;
//            }
//        }
//        Duration duration = Duration.between(start, Instant.now());
//        System.out.println(clicks / (duration.toSeconds()));
//    }
//
//    @Test
//    public void shouldDetermineYoochooseParameters() {
//        String filename = "yoochoose_sample.csv";
//        CsvDao csvDao = new CsvDao(filename);
//        List<SyntheticJourneySupplier.Row> rows = csvDao.readRows();
//        SyntheticJourneySupplier syntheticJourneySupplier = new SyntheticJourneySupplier(50000);
//        syntheticJourneySupplier.fit(rows);
//        Assertions.assertEquals(3.9278399999999984, syntheticJourneySupplier.getSessionLambda());
//
//        Assertions.assertEquals(3.2248428313441145E-5, syntheticJourneySupplier.getItemXMin());
//        Assertions.assertEquals(1.9090503828836067, syntheticJourneySupplier.getItemExponent());
//
//        List<Integer> sessionLengths = new ArrayList<>();
//        for (int i = 0; i < 100; i++) {
//            List<Long> items = syntheticJourneySupplier.get();
////            System.out.println(items.size());
//            sessionLengths.add(items.size());
//        }
//        Collections.sort(sessionLengths);
//
//        // Print the percentiles
//        System.out.println("p25: " + sessionLengths.get((int) (0.25 * sessionLengths.size())));
//        System.out.println("p50: " + sessionLengths.get((int) (0.5 * sessionLengths.size())));
//        System.out.println("p90: " + sessionLengths.get((int) (0.9 * sessionLengths.size())));
//        System.out.println("p100: " + sessionLengths.get(sessionLengths.size() - 1));
//    }
//
//    @Test
//    public void shouldDetermineBolcomParameters() {
//        String filename = "raw_click_lists.csv";
//        CsvDao csvDao = new CsvDao(filename);
//        List<SyntheticJourneySupplier.Row> rows = csvDao.readRows();
//        SyntheticJourneySupplier syntheticJourneySupplier = new SyntheticJourneySupplier(50000);
//        syntheticJourneySupplier.fit(rows);
//        Assertions.assertEquals(8.503983880226111E-5, syntheticJourneySupplier.getItemXMin());
//        Assertions.assertEquals(3.034168512358493, syntheticJourneySupplier.getItemExponent());
//    }
//
//    @Test
//    public void handleessionLengths() {
//        String filename = "yoochoose_sample.csv";
//        CsvDao csvDao = new CsvDao(filename);
//        List<SyntheticJourneySupplier.Row> rows = csvDao.readRows();
//        List<Double> sessionLengths = rows.stream().map(r -> r.getSessionId())
//                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting())).values().stream().map(r -> r.doubleValue()).collect(Collectors.toList());
//
//        Percentile pMethod = new Percentile();
//        double realP25 = pMethod.evaluate(sessionLengths.stream().mapToDouble(Double::doubleValue).toArray(), 25);
//        double realP50 = pMethod.evaluate(sessionLengths.stream().mapToDouble(Double::doubleValue).toArray(), 50);
//        double realP90 = pMethod.evaluate(sessionLengths.stream().mapToDouble(Double::doubleValue).toArray(), 90);
//        double realP99 = pMethod.evaluate(sessionLengths.stream().mapToDouble(Double::doubleValue).toArray(), 99);
//
//        // Estimate the power-law exponent using MLE
//        double alpha = estimatePowerLawAlpha(sessionLengths);
//
//        System.out.println("Estimated alpha: " + alpha);
//        List<Double> synthetic = new ArrayList<>();
//        for ( int i = 0 ; i < sessionLengths.size(); i++ ) {
//            synthetic.add(Math.ceil(generateSyntheticSessionLength(alpha)));
//        }
//        double synthP25 = pMethod.evaluate(synthetic.stream().mapToDouble(Double::doubleValue).toArray(), 25);
//        double synthP50 = pMethod.evaluate(synthetic.stream().mapToDouble(Double::doubleValue).toArray(), 50);
//        double synthP90 = pMethod.evaluate(synthetic.stream().mapToDouble(Double::doubleValue).toArray(), 90);
//        double synthP99 = pMethod.evaluate(synthetic.stream().mapToDouble(Double::doubleValue).toArray(), 99);
//        System.out.println("real p25:" + realP25 + " p50:" + realP50 + " p90:" + realP90 + " p99:" + realP99);
//        System.out.println("synth p25:" + synthP25 + " p50:" + synthP50 + " p90:" + synthP90 + " p99:" + synthP99);
//
//        // Calculate the shape parameter for Zipf distribution
//
//        alpha = estimateAlpha(realP25, realP50, realP90, realP99);
//        // Create a Zipf distribution
//// Instantiate a Pareto distribution with the estimated alpha
//        RealDistribution powerLawDistribution = new ParetoDistribution(alpha, 1.0);
//
//        List<Double> synthetic2 = new ArrayList<>();
//        for ( int i = 0 ; i < sessionLengths.size(); i++ ) {
//            synthetic2.add(powerLawDistribution.sample());
//        }
//        synthP25 = pMethod.evaluate(synthetic2.stream().mapToDouble(Double::doubleValue).toArray(), 25);
//        synthP50 = pMethod.evaluate(synthetic2.stream().mapToDouble(Double::doubleValue).toArray(), 50);
//        synthP90 = pMethod.evaluate(synthetic2.stream().mapToDouble(Double::doubleValue).toArray(), 90);
//        synthP99 = pMethod.evaluate(synthetic2.stream().mapToDouble(Double::doubleValue).toArray(), 99);
//        System.out.println("synth p25:" + synthP25 + " p50:" + synthP50 + " p90:" + synthP90 + " p99:" + synthP99);
//
//        double a = 0.10324840656752753;  // Shape parameter
//        double loc = 0.9999999999999999;  // Lower bound
//        double scale = 199.00000000000003;  // Scale parameter
//        int size = 10000;  // Nu
//
//        Random random = new Random();
//        double[] randomVariates = new double[size];
//        for (int i = 0; i < size; i++) {
//            double pow = Math.pow(random.nextDouble(), (1 / a));
//            double x = loc + (scale * pow);
//            randomVariates[i] = Math.min(Math.ceil(x), scale);
//        }
//        synthP25 = pMethod.evaluate(randomVariates, 25);
//        synthP50 = pMethod.evaluate(randomVariates, 50);
//        synthP90 = pMethod.evaluate(randomVariates, 90);
//        synthP99 = pMethod.evaluate(randomVariates, 99);
//        double synthP100 = pMethod.evaluate(randomVariates, 100);
//        System.out.println("synth p25:" + synthP25 + " p50:" + synthP50 + " p90:" + synthP90 + " p99:" + synthP99 + " p100:" + synthP100);
//
//
//        a = 0.040690006010909816;
//        loc = 0.9999999999999999;
//        scale = 2335.0000000000005;
//        double[] randomItemVariates = new double[10_000];
//        long now = System.currentTimeMillis();
//        for (int i = 0; i < 10_000_000; i++) {
//            double pow = Math.pow(random.nextDouble(), (1 / a));
//            double x = loc + (scale * pow);
//            double value = Math.min(Math.ceil(x), scale);
////            randomVariates[i] = value;
//        }
//        System.out.println((System.currentTimeMillis() - now) / 1000);
//        synthP25 = pMethod.evaluate(randomVariates, 25);
//        synthP50 = pMethod.evaluate(randomVariates, 50);
//        synthP90 = pMethod.evaluate(randomVariates, 90);
//        synthP99 = pMethod.evaluate(randomVariates, 99);
//        synthP100 = pMethod.evaluate(randomVariates, 100);
//        System.out.println("item synth p25:" + synthP25 + " p50:" + synthP50 + " p90:" + synthP90 + " p99:" + synthP99 + " p100:" + synthP100);
//
//
//
//
//    }
//
//
//
//    public static double estimateAlpha(double p25, double p50, double p90, double p99) {
//        double k = (3 * p50 - p25) / 2;
//        double alpha = 1 + 2 / (p99 - p25 - 4 * k / 3);
//
//        return alpha;
//    }
//
//
//    public static double generateSyntheticSessionLength(double alpha) {
//        double u = Math.random(); // Generate a random number between 0 and 1
//        return Math.pow(1.0 / (1.0 - u), 1.0 / (1.0 - alpha));
//    }
//
//    public static double estimatePowerLawAlpha(List<Double> data) {
//        // Sort the data in descending order
//        Collections.sort(data);
//        double[] reversedData = new double[data.size()];
//        for (int i = 0; i < data.size(); i++) {
//            reversedData[i] = data.get(data.size() - (i + 1));
//        }
//        double sumLogX = 0.0;
//        for (double x : reversedData) {
//            sumLogX += Math.log(x);
//        }
//
//        double n = data.size();
//        double alpha = 1 + n * sumLogX;
//
//        return alpha;
//    }
//}