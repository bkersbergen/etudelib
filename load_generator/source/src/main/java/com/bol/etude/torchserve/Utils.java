package com.bol.etude.torchserve;

import com.bol.etude.LoadGeneratorConfig;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import org.apache.http.Consts;
import org.apache.http.HttpResponse;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager;
import org.apache.http.util.EntityUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Utils {
    private static final Logger LOGGER = LogManager.getLogger(Utils.class);

    public static void sleep(long millis) {
        try {
            // Wait for some time to demonstrate threads
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static CloseableHttpClient createHttpClient(int qtyConnections) {
        PoolingHttpClientConnectionManager connManager
                = new PoolingHttpClientConnectionManager();

        connManager.setMaxTotal(qtyConnections);  // max number of open connections
        connManager.setDefaultMaxPerRoute(qtyConnections);  // max number of concurrent connections per route
        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectTimeout(2000)
                .setConnectionRequestTimeout(2000)
                .setSocketTimeout(2000).build();

        CloseableHttpClient httpClient = HttpClients.custom()
                .setConnectionManager(connManager)
                .setDefaultRequestConfig(requestConfig)
                .build();
        return httpClient;
    }

    public static String downloadModelInfo(CloseableHttpClient httpClient, TorchServeModelInfo mcc) throws IOException {
        String connectUri = "http://" + mcc.getServerIp() + ":8081/models/" + mcc.getServingModelName() + "/" + mcc.getServingModelVersion();
        HttpGet httpGet = new HttpGet(connectUri);
        HttpResponse response = httpClient.execute(httpGet);
        String json = EntityUtils.toString(response.getEntity(), Consts.UTF_8);
        return json;
    }

    public static int getQtyWorkersForModel(String connectUri) throws IOException {
        CloseableHttpClient httpClient = Utils.createHttpClient(1);
        HttpGet httpGet = new HttpGet(connectUri);
        HttpResponse response = httpClient.execute(httpGet);
        String json = EntityUtils.toString(response.getEntity(), Consts.UTF_8);
        httpClient.close();
        JsonArray workers = new Gson().fromJson(json, JsonArray.class).get(0).getAsJsonObject().get("workers").getAsJsonArray();
        return workers.size();
    }

    public static List<String> getDeployedModels(String serverIp) throws IOException {
        // returns a list of deployed models
        // Returns ["bert4rec_bolcom1m", "bert4rec_bolcom100k"]
        CloseableHttpClient httpClient = Utils.createHttpClient(1);
        String connectUri = "http://" + serverIp + ":8081/models/";
        HttpGet httpGet = new HttpGet(connectUri);
        HttpResponse response = httpClient.execute(httpGet);
        String json = EntityUtils.toString(response.getEntity(), Consts.UTF_8);
        httpClient.close();
        JsonArray models = new Gson().fromJson(json, JsonObject.class).get("models").getAsJsonArray();
        List<String> results = new ArrayList<>();
        for (JsonElement model : models) {
            for (Map.Entry<String, JsonElement> property : model.getAsJsonObject().entrySet()) {
                if (property.getKey().equals("modelName")) {
                    results.add(property.getValue().getAsString());
                }
            }
        }
        return results;
    }
}
