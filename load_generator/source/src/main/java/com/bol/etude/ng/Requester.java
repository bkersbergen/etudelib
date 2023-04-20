package com.bol.etude.ng;

import com.google.gson.Gson;
import org.apache.hc.client5.http.async.methods.SimpleHttpRequest;
import org.apache.hc.client5.http.async.methods.SimpleHttpResponse;
import org.apache.hc.client5.http.async.methods.SimpleRequestBuilder;
import org.apache.hc.client5.http.impl.async.CloseableHttpAsyncClient;
import org.apache.hc.client5.http.impl.async.HttpAsyncClients;
import org.apache.hc.core5.concurrent.FutureCallback;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.Phaser;

import static org.apache.hc.core5.http.ContentType.APPLICATION_JSON;

public class Requester<T> implements Closeable {
    private final Gson gson = new Gson();
    private static final String REQUEST_START = "request-start";
    private final CloseableHttpAsyncClient client = client();
    private final Phaser phaser = new Phaser();
    private final URI uri;
    private final Authenticator authenticator;

    Requester (@Nonnull URI uri, @Nullable Authenticator authenticator) {
        this.uri = uri;
        this.authenticator = authenticator;
    }

    Requester (URI uri) {
       this(uri, null);
    }

    void exec(T payload, Callback callback) {
        phaser.register();
        client.execute(post(payload), new FutureCallback<>() {
            @Override
            public void completed(SimpleHttpResponse response) {
                try {
                    Instant start = Instant.parse(response.getHeader(REQUEST_START).getValue());
                    String body = response.getBodyText();
                    int status = response.getCode();
                    Duration latency = Duration.between(start, Instant.now());

                    callback.callback(new Response(start, status, body, latency), null);
                } catch (Exception e) {
                    callback.callback(null, e);
                } finally {
                    phaser.arriveAndDeregister();
                }
            }

            @Override
            public void failed(Exception e) {
                phaser.arriveAndDeregister();
                callback.callback(null, e);
            }

            @Override
            public void cancelled() {
                phaser.arriveAndDeregister();
                callback.callback(null, null);
            }
        });
    }

    private SimpleHttpRequest post(T payload) {
        String body = gson.toJson(payload);
        return SimpleRequestBuilder.post()
                .setHeader("Content-Length", String.valueOf(body.length()))
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
        CloseableHttpAsyncClient client = HttpAsyncClients.custom()
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
                .build();
        client.start();
        return client;
    }

    interface Callback {
        void callback(Response success, Throwable failure);
    }

    static class Response {

        public Instant start;

        public final int status;
        public final String body;
        public  final Duration latency;

        Response(Instant start, int status, String body, Duration latency) {
            this.start = start;
            this.status = status;
            this.body = body;
            this.latency = latency;
        }
    }
}

