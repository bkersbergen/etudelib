package com.bol.etude.ng;

import com.bol.etude.dataproducer.Row;
import nl.peterbloem.powerlaws.Continuous;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.commons.math3.util.Pair;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class SyntheticJourneySupplier implements Supplier<List<Long>> {
    private int C = 0;
    private PoissonDistribution sessionLengthDistribution;
    private PolynomialSplineFunction itemFunction;

    private Random random = new Random();

    public SyntheticJourneySupplier(int C) {
        this.C = C;
        double lambda = 5.597568416279968;
        double xMin = 8.0E-5;
        double exponent = 3.650557039874508;
//        this.fit(lambda, xMin, exponent);
    }

    @Override
    public List<Long> get() {
        int sessionLength = Math.max(1, sessionLengthDistribution.sample());

        double rangeMin = itemFunction.getKnots()[0];
        double rangeMax = itemFunction.getKnots()[itemFunction.getN()];
        List<Long> result = new ArrayList<>();
        for (int i = 0 ; i < sessionLength; i++) {
            double randomValue = rangeMin + (rangeMax - rangeMin) * random.nextDouble();
            Long itemId = Math.round(itemFunction.value(randomValue));
            result.add(itemId);
        }
        return result;
    }

    public void fit(List<Row> rows) {
        double lambda = determineSessionLengthLambda(rows);
        Pair<Double, Double> xMinExponent = determineItemParameters(rows);
        double xMin = xMinExponent.getFirst();
        double exponent = xMinExponent.getSecond();

        this.fit(lambda, xMin, exponent);
    }

    private static Pair<Double, Double> determineItemParameters(List<Row> rows) {
        Collection<Integer> itemFreqs = rows.stream().map(r -> r.getItemId())
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting())).values().stream().map(r -> r.intValue()).collect(Collectors.toList());
        double itemFreqTotal = itemFreqs.stream().mapToDouble(f -> f.doubleValue()).sum();

        List<Double> itemProbas = itemFreqs.stream().map(f -> f/itemFreqTotal).collect(Collectors.toList());
        Collections.sort(itemProbas);
        Continuous fitted = Continuous.fit(itemProbas).fit();
        return new Pair<>(fitted.xMin(), fitted.exponent());

    }

    private static double determineSessionLengthLambda(List<Row> rows) {
        Collection<Integer> sessionLengths = rows.stream().map(r -> r.getSessionId())
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting())).values().stream().map(r -> r.intValue()).collect(Collectors.toList());

        Map<Integer, Long> frequency = sessionLengths.stream()
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));

        int n = sessionLengths.size();

        // find the optimal lambda using maximum likelihood estimation
        double lambda = 0;
        for (int x : frequency.keySet()) {
            long count = frequency.get(x);
            lambda += (double) count / n * x; // update lambda based on observed frequency and value
        }
        return lambda;
    }

    public void fit(double sessionLengthLambda, double xMin, double exponent) {
        System.out.println("sessionLengthLambda: " + sessionLengthLambda);
        System.out.println("xMin: " + xMin);
        System.out.println("exponent: " + exponent);
        this.sessionLengthDistribution = new PoissonDistribution(sessionLengthLambda);

        Continuous itemDistribution = new Continuous(xMin, exponent);

        // Generate probabilities for items of catalog size C
        List<Double> itemCdf = itemDistribution.generate(C);

        double total = itemCdf.stream().mapToDouble(f -> f.doubleValue()).sum();
        itemCdf = itemCdf.stream().map(f -> f/total).collect(Collectors.toList());
        Collections.sort(itemCdf);

        cumsum(itemCdf);

        double[] x = new double[itemCdf.size()];
        for(int i=0;i<x.length;i++) {
            x[i] = i;
        }

        double[] y = itemCdf.stream().mapToDouble(Double::doubleValue).toArray();
        LinearInterpolator interpolator = new LinearInterpolator();
        this.itemFunction = interpolator.interpolate(y, x);  // interpolate from proba to itemid
    }

    public void cumsum(List<Double> input) {
        double total = 0.0;
        for (int idx = 0 ; idx < input.size(); idx++) {
            total += input.get(idx);
            input.set(idx, total);
        }
    }
}
