package com.bol.etude.ng;

import com.bol.etude.generated.Report;
import com.bol.etude.ng.Journeys.Journey;

import java.io.File;
import java.net.URI;
import java.util.List;
import java.util.Random;
import java.util.stream.LongStream;

import static com.bol.etude.ng.Tester.rampThenHold;
import static java.time.Duration.ofSeconds;

public class Main {

    private static List<Long> items() {
        return LongStream.range(0, new Random().nextLong(1, 20)).boxed().toList();
    }

    public static void main(String[] args) {


        Requester<String> requester = new Requester<>(URI.create("https://httpbin.org/response-headers")/*, new GoogleBearerAuthenticator()*/);
        Persister<Report> persister = new DataFilePersister<>(new File("/tmp/etude/report.avro"), Report.class);

        try (persister; requester) {
            Journeys journeys = new Journeys(Main::items);
            Collector<Journey> collector = new Collector<>();

            rampThenHold(10, ofSeconds(10), ofSeconds(60), () -> {
                Journey journey = journeys.pull();

                requester.exec(journey.item(), (success, failure) -> {
                    if (success == null) {
                        collector.remove(journey);
                        System.out.println("item.err(journey = " + journey.uid + ", size = " + journey.size() + ", index = " + journey.index() + ")");
                    } else {
                        collector.add(journey, success);
                        System.out.println("item.ok(journey = " + journey.uid + ", size = " + journey.size() + ", index = " + journey.index() + ")");

                        if (journey.last()) {
                            List<com.bol.etude.ng.Requester.Response> responses = collector.remove(journey);
                            Report.Builder report = Report.newBuilder();
                            report.setContext("");
                            report.setRequests(journey.items());
                            report.setResponses(responses.stream().map(it -> {
                                com.bol.etude.generated.Response.Builder response = com.bol.etude.generated.Response.newBuilder();
                                response.setStatus(it.status);
                                response.setLatency(it.latency.toMillis());
                                response.setTimestamp(it.start.toEpochMilli());
                                response.setBody(it.body);
                                return response.build();
                            }).toList());

                            persister.accept(report.build());
                            System.out.println("journey.done(uuid = " + journey.uid + ", size = " + journey.size() + ")");
                        }

                        journeys.push(journey);
                    }
                });
            });
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }
}
