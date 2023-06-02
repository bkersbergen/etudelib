package com.bol.etude.ng;

import com.google.gson.annotations.SerializedName;

import java.util.List;

public class GoogleVertexResponse {
    public List<Prediction> predictions;

    static class Prediction {
        @SerializedName("nf")
        Timings timings;
        List<List<Long>> items;
    }

    static class Timings {
        @SerializedName("postprocess_ms")
        float postprocessing;

        @SerializedName("preprocess_ms")
        float preprocessing;

        @SerializedName("inference_ms")
        float inferencing;
    }
}
