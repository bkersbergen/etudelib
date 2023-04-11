package com.bol.etude.ng;

import com.bol.etude.generated.Interaction;
import com.bol.etude.generated.Report;
import com.bol.etude.ng.Journeys.Journey;
import com.bol.etude.ng.Requester.Response;
import com.google.common.base.Strings;
import com.google.gson.Gson;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import static com.bol.etude.ng.Tester.rampThenHold;
import static java.time.Duration.ofSeconds;

public class Main {

    private static final Gson gson = new Gson();

    public static void main(String[] args) throws InterruptedException {
        String endpoint = System.getenv("VERTX_ENDPOINT");
        System.out.println("ENV_VAR[VERTX_ENDPOINT] = '" + endpoint + "'");

        String catalog = System.getenv("CATALOG_SIZE");
        System.out.println("ENV_VAR[CATALOG_SIZE] = '" + catalog + "'");

        if (Strings.isNullOrEmpty(endpoint) || Strings.isNullOrEmpty(catalog)) {
            System.out.println("killing loadgen, env variables [VERTX_ENDPOINT, CATALOG_SIZE] are not both set");
            Thread.sleep(300_000);
            System.out.println("exit(1)");
            System.exit(1);
        }

//        String uri = "https://europe-west4-aiplatform.googleapis.com/v1/projects/bolcom-pro-reco-analytics-fcc/locations/europe-west4/endpoints/4775442882221834240:predict";
//        int catalogus = 1000000;

        Requester<GoogleVertxRequest> requester = new Requester<>(URI.create(endpoint), new GoogleBearerAuthenticator());
        Persister<Report> persister = new DataFilePersister<>(new File("/tmp/etude/report.avro"), Report.class);

        SyntheticJourneySupplier supplier = new SyntheticJourneySupplier(Integer.parseInt(catalog));
        double lambda = 5.597568416279968;
        double xMin = 8.0E-5;
        double exponent = 3.650557039874508;
        supplier.fit(lambda, xMin, exponent);

        try (persister; requester) {
            Journeys journeys = new Journeys(supplier);
            Collector<Journey> collector = new Collector<>();

            rampThenHold(200, ofSeconds(30), ofSeconds(600), (tick) -> {
                Journey journey = journeys.pull();

                requester.exec(new GoogleVertxRequest(journey.item()), (success, failure) -> {
                    if (success == null) {
                        collector.remove(journey);
                    } else {
                        collector.add(journey, success);

                        if (!journey.last()) {
                            journeys.push(journey);
                        } else {
                            Report report = report(journey, collector.remove(journey), gson);
                            persister.accept(report);
                        }
                    }
                });

                tick.doOnComplete(() -> {
                    try {
                        persister.flush();
                    } catch (IOException e) {
                        e.printStackTrace();
                        throw new RuntimeException(e);
                    }
                });
            });
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    private static Report report(Journey journey, List<Response> responses, Gson gson) {
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
        return report.build();
    }
}
