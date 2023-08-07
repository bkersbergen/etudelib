package com.bol.etude.ng;

import com.google.gson.annotations.SerializedName;

import java.util.List;

public class GoogleVertexResponse {
    @SerializedName("nf")
    Timings timings;
    List<List<Long>> items;

    static class Timings {
        @SerializedName("pre_ms")
        float preprocessing;

        @SerializedName("inf_ms")
        float inferencing;

        @SerializedName("post_ms")
        float postprocessing;

        @SerializedName("mname")
        String model_name;

        @SerializedName("mthreads")
        int model_thread_qty;

        @SerializedName("mdevice")
        String model_device;

    }
}
