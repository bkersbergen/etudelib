package com.bol.etude.ng;

import com.google.gson.annotations.SerializedName;

import java.util.List;

public class GoogleVertxResponse {
    public List<Prediction> predictions;

    static class Prediction {
        @SerializedName("nf")
        Timings timings;
        List<Long> items;
    }

    static class Timings {
        @SerializedName("postprocess_ms")
        float processing;

        @SerializedName("preprocess_ms")
        float preprocessing;

        @SerializedName("inference_ms")
        float inferencing;
    }
}