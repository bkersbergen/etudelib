package com.bol.etude.ng;

import com.google.gson.Gson;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MainTest {

    private String getJson() {
        String json = """
                         {                                                                                                                                                       │
                           "predictions": [                                                                                                                                      │
                             {                                                                                                                                                   │
                               "nf": {                                                                                                                                           │
                                 "postprocess_ms": 0.090100016677752137,                                                                                                         │
                                 "inference_ms": 0.0596430036239326,                                                                                                             │
                                 "model": "noop_bolcom_c1000000_t50_jitopt.pth",                                                                                                 │
                                 "device": "cpu",                                                                                                                                │
                                 "preprocess_ms": 0.16905498341657221                                                                                                            │
                               },                                                                                                                                                │
                               "items": [                                                                                                                                        │
                                 [                                                                                                                                               │
                                   20,                                                                                                                                           │
                                   19,                                                                                                                                           │
                                   18,                                                                                                                                           │
                                   17,                                                                                                                                           │
                                   16,                                                                                                                                           │
                                   15,                                                                                                                                           │
                                   14,                                                                                                                                           │
                                   13,                                                                                                                                           │
                                   12,                                                                                                                                           │
                                   11,                                                                                                                                           │
                                   10,                                                                                                                                           │
                                   9,                                                                                                                                            │
                                   8,                                                                                                                                            │
                                   7,                                                                                                                                            │
                                   6,                                                                                                                                            │
                                   5,                                                                                                                                            │
                                   4,                                                                                                                                            │
                                   3,                                                                                                                                            │
                                   2,                                                                                                                                            │
                                   1,                                                                                                                                            │
                                   0                                                                                                                                             │
                                 ]                                                                                                                                               │
                               ]                                                                                                                                                 │
                             }                                                                                                                                                   │
                           ],                                                                                                                                                    │
                           "deployedModelId": "5297539381601501184",                                                                                                             │
                           "model": "projects/1077776595046/locations/europe-west4/models/3871644324189962240",                                                                  │
                           "modelDisplayName": "noop_bolcom_c1000000_t50_jitopt",                                                                                                │
                           "modelVersionId": "4"                                                                                                                                 │
                         }                                                                                                                                                       │
                """;
        return json;
    }

    @Test
    public void test_happyflow() {
        String json = getJson();
        Gson gson = new Gson();
        GoogleVertexResponse vertex = gson.fromJson(json, GoogleVertexResponse.class);
        System.out.println(vertex);

    }


}