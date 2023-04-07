package com.bol.etude.ng;

import com.bol.etude.generated.Interaction;
import com.bol.etude.generated.Report;
import com.bol.etude.ng.Journeys.Journey;
import com.bol.etude.ng.Requester.Response;
import com.google.gson.Gson;

import java.io.File;
import java.net.URI;
import java.util.ArrayList;
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
        Gson gson = new Gson();
//        String uri = "https://httpbin.org/anything";
        String uri = "https://europe-west4-aiplatform.googleapis.com/v1/projects/bolcom-pro-reco-analytics-fcc/locations/europe-west4/endpoints/4775442882221834240:predict";
        Requester<GoogleVertxRequest> requester = new Requester<>(URI.create(uri), new GoogleBearerAuthenticator());

        Persister<Report> persister = new DataFilePersister<>(new File("/tmp/etude/report.avro"), Report.class);

        SyntheticJourneySupplier supplier = new SyntheticJourneySupplier(100);
        double lambda = 5.597568416279968;
        double xMin = 8.0E-5;
        double exponent = 3.650557039874508;
        supplier.fit(lambda, xMin, exponent);
        try (persister; requester) {
            Journeys journeys = new Journeys(Main::items);
            Collector<Journey> collector = new Collector<>();

            rampThenHold(200, ofSeconds(30), ofSeconds(600), () -> {
                Journey journey = journeys.pull();

                requester.exec(new GoogleVertxRequest(journey.item()), (success, failure) -> {
                    if (success == null) {
                        collector.remove(journey);
//                        System.out.println("item.err(journey = " + journey.uid + ", size = " + journey.size() + ", index = " + journey.index() + ")");
                    } else {
                        collector.add(journey, success);
//                        System.out.println("item.ok(journey = " + journey.uid + ", size = " + journey.size() + ", index = " + journey.index() + ")");

                        if (journey.last()) {
                            List<Response> responses = collector.remove(journey);
                            ArrayList<Interaction> interactions = new ArrayList<>();

                            for (int index = 0; index < journey.size(); index++) {
                                Response response = responses.get(index);

                                Interaction.Builder interaction = Interaction.newBuilder();
                                interaction.setTimestampEpochMillis(response.start.toEpochMilli());
                                interaction.setInput(journey.items().subList(0, index + 1));
                                interaction.setStatus(response.status);
                                interaction.setLatencyMillis(response.latency.toMillis());

                                GoogleVertxResponse vertx = gson.fromJson(response.body, GoogleVertxResponse.class);
                                GoogleVertxResponse.Prediction prediction = vertx.predictions.get(0);

                                interaction.setOutput(prediction.items);
                                interaction.setPreprocessingMillis(prediction.timings.preprocessing);
                                interaction.setInferencingMillis(prediction.timings.inferencing);
                                interaction.setProcessingMillis(prediction.timings.processing);

                                interactions.add(interaction.build());
                            }

                            Report.Builder report = Report.newBuilder();
                            report.setInteractions(interactions);

                            persister.accept(report.build());
//                            System.out.println("journey.done(uuid = " + journey.uid + ", size = " + journey.size() + ")");
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
