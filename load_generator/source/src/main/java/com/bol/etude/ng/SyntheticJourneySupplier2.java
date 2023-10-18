package com.bol.etude.ng;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class SyntheticJourneySupplier2 implements Supplier<List<Long>> {

    private final int C;
    private final Powerlaw sessionsDistribution;
    private final PolynomialSplineFunction itemFunction;
    private final double itemRangeMin;
    private final double itemRangeMax;
    private final Random random = new Random(42);

    public SyntheticJourneySupplier2(int C, double sessionsAlpha, double sessionsLoc, double sessionsScale, double itemsAlpha, double itemsLoc, double itemsScale) {
        this.C = C;
        this.sessionsDistribution = new Powerlaw(sessionsAlpha, sessionsLoc, sessionsScale);
        Powerlaw itemsDistribution = new Powerlaw(itemsAlpha, itemsLoc, itemsScale);

        // Generate probabilities for items of catalog size C
        List<Double> itemCdf = itemsDistribution.generate(C);

        double total = itemCdf.stream().mapToDouble(f -> f.doubleValue()).sum();
        itemCdf = itemCdf.stream().map(f -> f / total).collect(Collectors.toList());
        Collections.sort(itemCdf);

        cumsum(itemCdf);

        double[] x = new double[itemCdf.size()];
        for (int i = 0; i < x.length; i++) {
            x[i] = i;
        }

        double[] y = itemCdf.stream().mapToDouble(Double::doubleValue).toArray();
        LinearInterpolator interpolator = new LinearInterpolator();
        this.itemFunction = interpolator.interpolate(y, x);  // interpolate from proba to itemid
        this.itemRangeMin = itemFunction.getKnots()[0];
        this.itemRangeMax = itemFunction.getKnots()[itemFunction.getN()];
    }

    @Override
    public List<Long> get() {
        int sessionLength = Math.max(1, (int) Math.round(sessionsDistribution.generate()));
        List<Long> result = new ArrayList<>();
        for (int i = 0; i < sessionLength; i++) {
            double randomValue = (itemRangeMax - itemRangeMin) * random.nextDouble() + itemRangeMin;
            Long itemId = Math.round(itemFunction.value(randomValue));
            result.add(itemId);
        }
        return result;
    }

    private void cumsum(List<Double> input) {
        double total = 0.0;
        for (int idx = 0; idx < input.size(); idx++) {
            total += input.get(idx);
            input.set(idx, total);
        }
    }

}
