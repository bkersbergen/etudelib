package com.bol.etude.ng;

import com.bol.etude.generated.Interaction;
import com.bol.etude.generated.Report;
import com.bol.etude.ng.Journeys.Journey;
import com.bol.etude.ng.Requester.Response;
import com.google.cloud.storage.Bucket;
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageOptions;
import com.google.common.base.Strings;
import com.google.gson.Gson;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

import static com.bol.etude.ng.Tester.rampThenHold;
import static java.time.Duration.ofSeconds;

public class Main {

    private static final Gson gson = new Gson();

    public static void main(String[] args) throws InterruptedException, IOException {
        String endpoint_arg = System.getenv("VERTEX_ENDPOINT");
        System.out.println("ENV_VAR[VERTEX_ENDPOINT] = '" + endpoint_arg + "'");

        String catalog_size_arg = System.getenv("CATALOG_SIZE");
        System.out.println("ENV_VAR[CATALOG_SIZE] = '" + catalog_size_arg + "'");

        String report_location_arg = System.getenv("REPORT_LOCATION");
        System.out.println("ENV_VAR[REPORT_LOCATION] = '" + report_location_arg + "'");

        if (Strings.isNullOrEmpty(endpoint_arg) || Strings.isNullOrEmpty(catalog_size_arg) || Strings.isNullOrEmpty(report_location_arg)) {
            System.out.println("killing loadgen, env variables [VERTEX_ENDPOINT, CATALOG_SIZE, RUNTIME, REPORT_LOCATION] are not all set");
            Thread.sleep(300_000);
            System.out.println("exit(1)");
            System.exit(1);
        }

//        String endpoint = "https://europe-west4-aiplatform.googleapis.com/v1/projects/bolcom-pro-reco-analytics-fcc/locations/europe-west4/endpoints/4775442882221834240:predict";
//        int catalogus = 1000000;
//        String report = gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/04-18-2023-${model}-bolcom-c${size}-t50-${runtime}

        URI endpoint = URI.create(endpoint_arg);
        File temporary = new File("/tmp/etude/report.avro");
        Journeys journeys = createSyntheticJourneys(Integer.parseInt(catalog_size_arg));

        executeTestScenario(endpoint, temporary, journeys);
        writeReportToStorage(temporary, report_location_arg);
    }

    private static Journeys createSyntheticJourneys(int size) {
        SyntheticJourneySupplier journeys = new SyntheticJourneySupplier(size);
        journeys.fit(5.597568416279968,8.0E-5, 3.650557039874508);
        return new Journeys(journeys);
    }

    private static void executeTestScenario(URI endpoint, File temporary, Journeys journeys) {
        Requester<GoogleVertxRequest> requester = new Requester<>(endpoint, new GoogleBearerAuthenticator());
        Persister<Report> persister = new DataFilePersister<>(temporary, Report.class);
        Collector<Journey> collector = new Collector<>();

        try (persister; requester) {
            rampThenHold(200, ofSeconds(300), ofSeconds(600), (tick) -> {
                Journey journey = journeys.pull();

                requester.exec(new GoogleVertxRequest(journey.item()), (success, failure) -> {
                    if (success == null) {
                        collector.remove(journey);
                    } else {
                        collector.add(journey, success);

                        if (!journey.last()) {
                            journeys.push(journey);
                        } else {
                            Report report = buildJourneyReport(journey, collector.remove(journey), gson);
                            persister.accept(report);
                        }
                    }
                });

                tick.doOnComplete(() -> {
                    System.out.println(tick);
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

    private static Report buildJourneyReport(Journey journey, List<Response> responses, Gson gson) {
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

    private static void writeReportToStorage(File temporary, String permanent) throws IOException {
        if (permanent.startsWith("gs://")) {
            Storage storage = StorageOptions.getDefaultInstance().getService();
            URI uri = URI.create(permanent);
            Bucket bucket = storage.get(uri.getHost());
            bucket.create(uri.getPath(), Files.newInputStream(temporary.toPath()));
        } else {
            Files.copy(temporary.toPath(), new File(permanent).toPath());
        }
    }
}
