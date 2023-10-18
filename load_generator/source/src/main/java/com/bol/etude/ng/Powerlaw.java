package com.bol.etude.ng;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Powerlaw {


    private final double alpha;
    private final double loc;
    private final double scale;
    private final Random random = new Random(42);

    public Powerlaw(double alpha, double loc, double scale) {
        this.alpha = alpha;
        this.loc = loc;
        this.scale = scale;
    }

    public double generate() {
        double pow = Math.pow(random.nextDouble(), (1 / alpha));
        double x = loc + (scale * pow);
        return Math.min(x, scale);
    }

    public List<Double> generate(int n) {
        List<Double> points = new ArrayList<>(n);
        for (int i = 0; i < n; i++)
            points.add(generate());

        return points;
    }

    public double getAlpha() {
        return alpha;
    }

    public double getLoc() {
        return loc;
    }

    public double getScale() {
        return scale;
    }
}
