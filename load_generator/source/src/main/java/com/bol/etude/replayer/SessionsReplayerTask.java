package com.bol.etude.replayer;

import com.bol.etude.LoadGeneratorConfig;
import com.bol.etude.dataproducer.UserSession;
import com.bol.etude.generated.PyTorchResult;
import com.google.common.util.concurrent.RateLimiter;
import org.apache.http.Consts;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.util.EntityUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;

public class SessionsReplayerTask implements Runnable {
    private final BlockingQueue<UserSession> sessionUpdatesQueue;
    private final ConcurrentLinkedQueue<PyTorchResult> predictionOutputQueue;
    // https://github.com/ok2c/httpclient-benchmark/wiki
    private final CloseableHttpClient httpClient;
    private final RateLimiter rateLimiter;
    private final LoadGeneratorConfig config;
    private volatile boolean keepRunning;
    private static String connectUri = null;

    private static final Logger LOGGER = LogManager.getLogger(SessionsReplayerTask.class);

    /**
     * @param sessionUpdatesQueue   queue that contains the full session of this user
     * @param predictionOutputQueue queue where we put the results
     * @param httpClient            the http client used to connect to the prediction service
     * @param rateLimiter
     */
    public SessionsReplayerTask(BlockingQueue<UserSession> sessionUpdatesQueue, ConcurrentLinkedQueue<PyTorchResult> predictionOutputQueue, CloseableHttpClient httpClient, LoadGeneratorConfig config, RateLimiter rateLimiter) {
        this.sessionUpdatesQueue = sessionUpdatesQueue;
        this.predictionOutputQueue = predictionOutputQueue;
        this.httpClient = httpClient;
        this.keepRunning = true;
        this.rateLimiter = rateLimiter;
        this.config = config;
    }

    @Override
    public void run() {
        while (keepRunning) {
            try {
                UserSession userSession = sessionUpdatesQueue.poll(1, TimeUnit.SECONDS);
                if (userSession != null) {
                    replaySession(userSession);
                }
            } catch (InterruptedException e) {
                LOGGER.error(e);
            } catch (IOException e) {
                LOGGER.error(e);
            }
        }
        LOGGER.debug("Stopped");
    }

    private void replaySession(UserSession userSession) throws IOException {
        for (int i = 0; i < userSession.getItems().size(); i++) {
            if (keepRunning) {
                List<Long> evolvingSessionItems = userSession.getItems().subList(0, i + 1);
                List<Long> nextSessionItems = userSession.getItems().subList(i, userSession.getItems().size());
                String json = "{\"instances\": [{\"context\":" + evolvingSessionItems + "}],\"parameters\": [{\"runtime\":  \""+config.runtime+"\"}]}";
                HttpPost httpPost = createHttpPost(json);
                rateLimiter.acquire(); // may wait
                Instant start = Instant.now();
                HttpResponse response = httpClient.execute(httpPost);
                Duration duration = Duration.between(start, Instant.now());
                String content = EntityUtils.toString(response.getEntity(), Consts.UTF_8);
                PyTorchResult.Builder prb = PyTorchResult.newBuilder();
                prb.setContext(json);
                prb.setResponse(content);
                prb.setHttpStatusCode(response.getStatusLine().getStatusCode());
                prb.setLatency(duration.toMillis());
                prb.setEvolvingSessionItems(evolvingSessionItems);
                prb.setNextSessionItems(nextSessionItems);
                prb.setEventTimestamp(start.toEpochMilli());
                prb.setIsEvaluationData(userSession.isEvaluationData());
                predictionOutputQueue.add(prb.build());
            }
        }
    }

    public HttpPost createHttpPost(String json) throws UnsupportedEncodingException {
        HttpPost httpPost = new HttpPost(this.config.connectUri);
        httpPost.setEntity(new StringEntity(json));
        for (Map.Entry<String, String> headerEntry : this.config.httpHeader.entrySet()) {
            httpPost.setHeader(headerEntry.getKey(), headerEntry.getValue());
        }
        return httpPost;
    }

    public void stop() {
        keepRunning = false;
        LOGGER.debug("stop()");
    }
}
