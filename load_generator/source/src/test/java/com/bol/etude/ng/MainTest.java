package com.bol.etude.ng;

import com.google.gson.Gson;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MainTest {

    private String getJson() {
        String json = """
{
  "predictions": [
    {
      "nf": {
        "preprocess_ms": 0.44839101610705262,
        "device": "cpu",
        "postprocess_ms": 0.1220889971591532,
        "inference_ms": 1.107118994696066,
        "model": "noop_bolcom_c1000000_t50_jitopt.pth"
      },
      "items": [
        [
          20,
          19,
          18,
          17,
          16,
          15,
          14,
          13,
          12,
          11,
          10,
          9,
          8,
          7,
          6,
          5,
          4,
          3,
          2,
          1,
          0
        ]
      ]
    }
  ],
  "deployedModelId": "4913607513368166400",
  "model": "projects/1077776595046/locations/europe-west4/models/3871644324189962240",
  "modelDisplayName": "noop_bolcom_c1000000_t50_jitopt",
  "modelVersionId": "2"
}        """;
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