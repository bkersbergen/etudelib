package com.bol.etude.ng;

import com.google.gson.Gson;
import org.apache.hc.client5.http.async.methods.SimpleHttpResponse;
import org.apache.hc.client5.http.async.methods.SimpleResponseConsumer;
import org.apache.hc.client5.http.impl.async.CloseableHttpAsyncClient;
import org.apache.hc.client5.http.impl.async.HttpAsyncClients;
import org.apache.hc.core5.concurrent.FutureCallback;
import org.apache.hc.core5.http.nio.AsyncEntityProducer;
import org.apache.hc.core5.http.nio.AsyncRequestProducer;
import org.apache.hc.core5.http.nio.DataStreamChannel;
import org.apache.hc.core5.http.nio.entity.BasicAsyncEntityProducer;
import org.apache.hc.core5.http.nio.support.AsyncRequestBuilder;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.time.Instant;
import java.util.Set;
import java.util.concurrent.Phaser;
import java.util.function.Supplier;

import static org.apache.hc.core5.http.ContentType.APPLICATION_JSON;

public class Requester<T> implements Closeable {

    private static final Gson gson = new Gson();
    private static final String REQUEST_START = "request-start";
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

    void exec(Supplier<Journeys.Journey> factory, Callback callback) {
        phaser.register();
        Payloader payloader = new Payloader(factory);
        client.execute(post(payloader), SimpleResponseConsumer.create(), new FutureCallback<>() {
            @Override
            public void completed(SimpleHttpResponse response) {
                try {
                    Instant start = Instant.parse(response.getHeader(REQUEST_START).getValue());
                    String body = response.getBodyText();
                    int status = response.getCode();
                    Duration latency = Duration.between(start, Instant.now());
                    callback.callback(payloader.journey(), new Response(start, status, body, latency), null);
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
                callback.callback(payloader.journey(), null, e);
            }

            @Override
            public void cancelled() {
                phaser.arriveAndDeregister();
                callback.callback(payloader.journey(), null, null);
            }
        });
    }

    private AsyncRequestProducer post(Payloader payload) {
        return AsyncRequestBuilder.post(uri)
                .setEntity(payload)
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
                    request.setHeader("Content-Length", entity.getContentLength());
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
        void callback(Journeys.Journey journey, Response success, Throwable failure);
    }

    static class Response {

        public Instant start;

        public final int status;
        public final String body;
        public final Duration latency;

        Response(Instant start, int status, String body, Duration latency) {
            this.start = start;
            this.status = status;
            this.body = body;
            this.latency = latency;
        }
    }

    static class Payloader implements AsyncEntityProducer {

        private final Supplier<Journeys.Journey> factory;

        private Journeys.Journey journey =  null;


        private BasicAsyncEntityProducer delegate = null;

        Payloader(Supplier<Journeys.Journey> factory) {
            this.factory = factory;
        }

        public Journeys.Journey journey() {
            return journey;
        }

        private BasicAsyncEntityProducer delegate() {
            if (delegate == null) {
//                System.out.println("Journey.gen(thread = '" + Thread.currentThread().getName() + "')");
                journey = factory.get();
                GoogleVertexRequest request = new GoogleVertexRequest(journey.item());
                delegate = new BasicAsyncEntityProducer(gson.toJson(request), APPLICATION_JSON);
            }
            return delegate;
        }

        @Override
        public boolean isRepeatable() {
            return delegate().isRepeatable();
        }

        @Override
        public void failed(Exception cause) {
            cause.printStackTrace();
        }

        @Override
        public long getContentLength() {
            return delegate().getContentLength();
        }

        @Override
        public String getContentType() {
            return delegate().getContentType();
        }

        @Override
        public String getContentEncoding() {
            return delegate().getContentEncoding();
        }

        @Override
        public boolean isChunked() {
            return delegate().isChunked();
        }

        @Override
        public Set<String> getTrailerNames() {
            return delegate().getTrailerNames();
        }

        @Override
        public int available() {
            return delegate().available();
        }

        @Override
        public void produce(DataStreamChannel channel) throws IOException {
            delegate().produce(channel);
        }

        @Override
        public void releaseResources() {
            delegate().releaseResources();
        }
    }
}

