package com.bol.etude.ng;

import com.google.gson.Gson;
import org.apache.hc.client5.http.async.methods.SimpleHttpRequest;
import org.apache.hc.client5.http.async.methods.SimpleHttpResponse;
import org.apache.hc.client5.http.async.methods.SimpleRequestBuilder;
import org.apache.hc.client5.http.impl.async.CloseableHttpAsyncClient;
import org.apache.hc.client5.http.impl.async.HttpAsyncClients;
import org.apache.hc.client5.http.impl.nio.PoolingAsyncClientConnectionManager;
import org.apache.hc.core5.concurrent.FutureCallback;
import org.apache.hc.core5.http.Header;
import org.apache.hc.core5.reactor.IOReactorConfig;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.concurrent.Phaser;

import static org.apache.hc.core5.http.ContentType.APPLICATION_JSON;

public class Requester<T> implements Closeable {

    private static final Gson gson = new Gson();
    private static final String REQUEST_START = "request-start";
    private static final String X_SERVER_LATENCY_MS = "x-server-latency-ms";
    private final CloseableHttpAsyncClient client = client();
    private final Phaser phaser = new Phaser();
    private final URI uri;
    private final Authenticator authenticator;

    Requester(@Nonnull URI uri, @Nullable Authenticator authenticator) {
        this.uri = uri;
        this.authenticator = authenticator;
    }

    Requester(URI uri) {
        this(uri, null);
    }

    void exec(Journeys.Journey journey, Callback callback) {
        phaser.register();

        client.execute(post(journey), new FutureCallback<>() {
            @Override
            public void completed(SimpleHttpResponse response) {
                try {
                    Instant start = Instant.parse(response.getHeader(REQUEST_START).getValue());
                    String body = response.getBodyText();
                    int status = response.getCode();
                    Duration totalLatency = Duration.between(start, Instant.now());

                    Header serverLatencyHeader = response.getHeader(X_SERVER_LATENCY_MS);
                    Duration serverSideLatency = Duration.ofMillis(serverLatencyHeader != null ? Integer.parseInt(serverLatencyHeader.getValue()) : 0);

                    callback.callback(journey, new Response(start, status, body, totalLatency, serverSideLatency), null);
                } catch (Exception e) {
                    System.out.println("Requester.exec().err['" + e.getClass().getSimpleName() + "']");
                    e.printStackTrace();
                } finally {
                    phaser.arriveAndDeregister();
                }
            }

            @Override
            public void failed(Exception e) {
                phaser.arriveAndDeregister();
                callback.callback(journey, null, e);
            }

            @Override
            public void cancelled() {
                phaser.arriveAndDeregister();
                callback.callback(journey, null, null);
            }
        });
    }


    private SimpleHttpRequest post(Journeys.Journey journey) {
        List<Long> evolvingSession = journey.item();
        String body = gson.toJson(new GoogleVertexRequest(evolvingSession));
        return SimpleRequestBuilder.post()
//                .setHeader("Content-Length", String.valueOf(body.length()))
                .setBody(body, APPLICATION_JSON)
                .setUri(uri)
                .build();
    }

    @Override
    public void close() throws IOException {
        phaser.arriveAndAwaitAdvance();
        client.close();
    }

    private CloseableHttpAsyncClient client() {
        PoolingAsyncClientConnectionManager connections = new PoolingAsyncClientConnectionManager();
        connections.setMaxTotal(500);
        connections.setDefaultMaxPerRoute(500);

        CloseableHttpAsyncClient client = HttpAsyncClients.custom()
//                .setVersionPolicy(FORCE_HTTP_1)
                .addRequestInterceptorFirst((request, entity, context) -> {
                    if (authenticator != null) {
                        request.setHeader("Authorization", "Bearer " + authenticator.token());
                    }
                    context.setAttribute(REQUEST_START, Instant.now());
                })
                .addResponseInterceptorLast((response, entity, context) -> {
                    Instant start = (Instant) context.getAttribute(REQUEST_START);
                    response.setHeader(REQUEST_START, start);
                })
                .setIOReactorConfig(IOReactorConfig.custom().setIoThreadCount(2).build())
                .setConnectionManager(connections)
                .build();
        client.start();
        return client;
    }

    interface Callback {
        void callback(Journeys.Journey journey, Response success, Throwable failure);
    }

    static class Response {

        public Instant start;

        public final int status;
        public final String body;
        public final Duration latency;
        public final Duration serverSideLatency;

        Response(Instant start, int status, String body, Duration latency, Duration serverSideLatency) {
            this.start = start;
            this.status = status;
            this.body = body;
            this.latency = latency;
            this.serverSideLatency = serverSideLatency;
        }
    }
}

