import com.google.api.client.util.Lists;
import org.apache.http.Consts;
import org.apache.http.HttpResponse;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager;
import org.apache.http.util.EntityUtils;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Scanner;

public class CustomUsingGoogle {

    @Test
    public void doitourselves() throws IOException {

        String project = "bolcom-pro-reco-analytics-fcc";
        String endpointId = "2765993023484461056";

        List<Long> evolvingSessionItems = Lists.newArrayList();
        evolvingSessionItems.add(9300000080086393L);
        String runtime = "onnx";
        int qtyConnections = 1;
        PoolingHttpClientConnectionManager connManager
                = new PoolingHttpClientConnectionManager();

        connManager.setMaxTotal(qtyConnections);  // max number of open connections
        connManager.setDefaultMaxPerRoute(qtyConnections);  // max number of concurrent connections per route
        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectTimeout(20000)
                .setConnectionRequestTimeout(20000)
                .setSocketTimeout(20000).build();

        CloseableHttpClient httpClient = HttpClients.custom()
                .setConnectionManager(connManager)
                .setDefaultRequestConfig(requestConfig)
                .build();

        String json2 = "{\"instances\": [{\"context\":" + evolvingSessionItems + "}],\"parameters\": [{\"runtime\":  \""+runtime+"\"}]}";
        String connectUri = String.format("https://europe-west4-aiplatform.googleapis.com/v1/projects/%s/locations/europe-west4/endpoints/%s:predict",project, endpointId);
        System.out.println(json2);
        HttpPost httpPost = new HttpPost(connectUri);
        httpPost.setEntity(new StringEntity(json2));

//        httpPost.setHeader("Accept", "application/json");
        httpPost.setHeader("Content-type", "application/json");
//        httpPost.setHeader("Authorization", "Bearer " + token);
        Instant start = Instant.now();
        HttpResponse response = httpClient.execute(httpPost);
        Duration duration = Duration.between(start, Instant.now());
        String content = EntityUtils.toString(response.getEntity(), Consts.UTF_8);
        System.out.println(content);
        System.out.println("Duration (ms): " + duration.toMillis());
    }
}