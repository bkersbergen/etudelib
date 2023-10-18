//package com.bol.etude.ng;
//
//import nl.peterbloem.powerlaws.Continuous;
//import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
//import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
//import org.apache.commons.math3.distribution.PoissonDistribution;
//import org.apache.commons.math3.util.Pair;
//
//import java.util.*;
//import java.util.function.Function;
//import java.util.function.Supplier;
//import java.util.stream.Collectors;
//
//public class SyntheticJourneySupplier implements Supplier<List<Long>> {
//    private int C = 0;
//    private PoissonDistribution sessionLengthDistribution;
//    private PolynomialSplineFunction itemFunction;
//    private double rangeMin;
//
//    private double rangeMax;
//    private Random random = new Random();
//
//    private double sessionLambda;
//
//    private double itemXMin;
//    private double itemExponent;
//
//    public SyntheticJourneySupplier(int C) {
//        this.C = C;
//    }
//
//    @Override
//    public List<Long> get() {
//        int sessionLength = Math.max(1, sessionLengthDistribution.sample());
//        assert this.rangeMin != -1.0;
//        assert this.rangeMax != -1.0;
//
//        List<Long> result = new ArrayList<>();
//        for (int i = 0; i < sessionLength; i++) {
//            double randomValue = this.rangeMax * random.nextDouble();
//            Long itemId = Math.round(itemFunction.value(randomValue));
//            result.add(itemId);
//        }
//        return result;
//    }
//
//    public void fit(List<Row> rows) {
//        double sessionLambda = determineSessionLengthLambda(rows);
//        Pair<Double, Double> xMinExponent = determineItemParameters(rows);
//        double xMin = xMinExponent.getFirst();
//        double exponent = xMinExponent.getSecond();
//
//        this.fit(sessionLambda, xMin, exponent);
//    }
//
//    private static Pair<Double, Double> determineItemParameters(List<Row> rows) {
//        Collection<Integer> itemFreqs = rows.stream().map(r -> r.getItemId())
//                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting())).values().stream().map(r -> r.intValue()).collect(Collectors.toList());
//        double itemFreqTotal = itemFreqs.stream().mapToDouble(f -> f.doubleValue()).sum();
//
//        List<Double> itemProbas = itemFreqs.stream().map(f -> f / itemFreqTotal).collect(Collectors.toList());
//        Collections.sort(itemProbas);
//        Continuous fitted = Continuous.fit(itemProbas).fit();
//        return new Pair<>(fitted.xMin(), fitted.exponent());
//
//    }
//
//    private static double determineSessionLengthLambda(List<Row> rows) {
//        Collection<Integer> sessionLengths = rows.stream().map(r -> r.getSessionId())
//                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting())).values().stream().map(r -> r.intValue()).collect(Collectors.toList());
//
//        Map<Integer, Long> frequency = sessionLengths.stream()
//                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
//
//        int n = sessionLengths.size();
//
//        // find the optimal lambda using maximum likelihood estimation
//        double lambda = 0;
//        for (int x : frequency.keySet()) {
//            long count = frequency.get(x);
//            lambda += (double) count / n * x; // update lambda based on observed frequency and value
//        }
//        return lambda;
//    }
//
//
//    public void fit(double sessionLambda, double itemXMin, double itemExponent) {
//        System.out.println("SyntheticJourneys.fit(xMin ='" + itemXMin + "', lambda = '" + sessionLambda + "', exponent = '" + itemExponent + "')");
//
//        this.sessionLambda = sessionLambda;
//        this.itemXMin = itemXMin;
//        this.itemExponent = itemExponent;
//        this.sessionLengthDistribution = new PoissonDistribution(sessionLambda);
//
//        Continuous itemDistribution = new Continuous(itemXMin, itemExponent);
//
//
//        // Generate probabilities for items of catalog size C
//        List<Double> itemCdf = itemDistribution.generate(C);
//
//        double total = itemCdf.stream().mapToDouble(f -> f.doubleValue()).sum();
//        itemCdf = itemCdf.stream().map(f -> f / total).collect(Collectors.toList());
//        Collections.sort(itemCdf);
//
//        cumsum(itemCdf);
//
//        double[] x = new double[itemCdf.size()];
//        for (int i = 0; i < x.length; i++) {
//            x[i] = i;
//        }
//
//        double[] y = itemCdf.stream().mapToDouble(Double::doubleValue).toArray();
//        LinearInterpolator interpolator = new LinearInterpolator();
//        this.itemFunction = interpolator.interpolate(y, x);  // interpolate from proba to itemid
//        this.rangeMin = itemFunction.getKnots()[0];
//        this.rangeMax = itemFunction.getKnots()[itemFunction.getN()];
//
//    }
//
//
//    public void cumsum(List<Double> input) {
//        double total = 0.0;
//        for (int idx = 0; idx < input.size(); idx++) {
//            total += input.get(idx);
//            input.set(idx, total);
//        }
//    }
//
//    public double getSessionLambda() {
//        return sessionLambda;
//    }
//
//
//    public double getItemXMin() {
//        return itemXMin;
//    }
//
//    public double getItemExponent() {
//        return itemExponent;
//    }
//
//    public static class Row {
//
//        public int sessionId;
//        public long itemId;
//        public int time;
//
//        public Row(Integer sessionId, long itemId, int time) {
//            this.sessionId = sessionId;
//            this.itemId = itemId;
//            this.time = time;
//        }
//
//        public int getSessionId() {
//            return sessionId;
//        }
//
//        public long getItemId() {
//            return itemId;
//        }
//
//        public int getTime() {
//            return time;
//        }
//
//        public String toString() {
//            return "SessionId: " + this.sessionId + " ItemId: " + this.itemId + " time: " + this.time;
//        }
//
//
//        @Override
//        public boolean equals(Object o) {
//            if (this == o) return true;
//            if (o == null || getClass() != o.getClass()) return false;
//            Row row = (Row) o;
//            return sessionId == row.sessionId &&
//                    itemId == row.itemId &&
//                    time == row.time;
//        }
//
//        @Override
//        public int hashCode() {
//            return Objects.hash(sessionId, itemId, time);
//        }
//
//    }
//}
