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
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.bol.etude.ng.Tester.rampWithBackPressure;
import static java.time.Duration.ofSeconds;

public class Main {

    private static final Gson gson = new Gson();

    public static void main(String[] args) throws InterruptedException, IOException {
        String endpoint_arg = System.getenv("VERTEX_ENDPOINT");
//        endpoint_arg = "https://europe-west4-aiplatform.googleapis.com/v1/projects/1077776595046/locations/europe-west4/endpoints/4251142961540104192:predict";
        System.out.println("ENV_VAR[VERTEX_ENDPOINT] = '" + endpoint_arg + "'");

        String catalog_size_arg = System.getenv("CATALOG_SIZE");
//        catalog_size_arg = "1000000";
        System.out.println("ENV_VAR[CATALOG_SIZE] = '" + catalog_size_arg + "'");

        String report_location_arg = System.getenv("REPORT_LOCATION");
//        report_location_arg = "/tmp/etude.avro"; // "gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/xxx.avro";
        System.out.println("ENV_VAR[REPORT_LOCATION] = '" + report_location_arg + "'");

        if (Strings.isNullOrEmpty(endpoint_arg) || Strings.isNullOrEmpty(catalog_size_arg) || Strings.isNullOrEmpty(report_location_arg)) {
            System.out.println("env variables [VERTEX_ENDPOINT, CATALOG_SIZE, RUNTIME, REPORT_LOCATION] are required.");
            Thread.sleep(300_000);
            System.exit(1);
        }

        try {
            URI endpoint = URI.create(endpoint_arg);
            File temporary = new File("/tmp/etude/report.avro");
            Journeys journeys = createSyntheticJourneys(Integer.parseInt(catalog_size_arg));

            executeTestScenario(endpoint, temporary, journeys);
            writeReportToStorage(temporary, report_location_arg);

            System.out.println("Test.done().ok");
            System.exit(0);
        } catch (Throwable t) {
            System.out.println("Test.done().err");
            Thread.sleep(300_000);
            System.exit(1);
        }
    }

    private static Journeys createSyntheticJourneys(int size) {
        SyntheticJourneySupplier journeys = new SyntheticJourneySupplier(size);
        journeys.fit(5.597568416279968, 8.0E-5, 3.650557039874508);
        return new Journeys(journeys);
    }

    private static void executeTestScenario(URI endpoint, File temporary, Journeys journeys) {
        Requester<GoogleVertexRequest> requester = new Requester<>(endpoint, new GoogleBearerAuthenticator());
        Persister<Report> persister = new DataFilePersister<>(temporary, Report.class);
        Collector<Journey> collector = new Collector<>();

        try (persister; requester) {
            rampWithBackPressure(500, ofSeconds(500), (request) -> {
                request.start();
                Journey journey = journeys.pull();

                requester.exec(new GoogleVertexRequest(journey.item()), (success, failure) -> {
                    request.complete();
                    Requester.Response response = success == null
                            ? new Requester.Response(Instant.EPOCH, 500, "", Duration.ofMillis(-1))
                            : success;

                    collector.add(journey, response);

                    if (!journey.last()) {
                        journeys.push(journey);
                    } else {
                        Report report = buildJourneyReport(journey, collector.remove(journey), gson);
                        persister.accept(report);
                    }
                });

                request.doOnTickStart(() -> {
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
            interaction.setLatencyMillis(response.latency.toMillis());

            try {
                GoogleVertexResponse vertex = gson.fromJson(response.body, GoogleVertexResponse.class);
                GoogleVertexResponse.Prediction prediction = vertex.predictions.get(0);
                interaction.setOutput(prediction.items);
                interaction.setPreprocessingMillis(prediction.timings.preprocessing);
                interaction.setInferencingMillis(prediction.timings.inferencing);
                interaction.setProcessingMillis(prediction.timings.processing);
            } catch (Throwable t) {
                t.printStackTrace();
                interaction.setStatus(500);
                interaction.setOutput(Collections.emptyList());
                interaction.setPreprocessingMillis(-1);
                interaction.setInferencingMillis(-1);
                interaction.setProcessingMillis(-1);
            }

            interactions.add(interaction.build());
        }

        Report.Builder report = Report.newBuilder();
        report.setInteractions(interactions);
        return report.build();
    }

    private static void writeReportToStorage(File temporary, String permanent) {
        try {
            System.out.println("Storage.write(uri = '" + permanent + "')");
            if (permanent.startsWith("gs://")) {
                Storage storage = StorageOptions.getDefaultInstance().getService();
                URI uri = URI.create(permanent);
                Bucket bucket = storage.get(uri.getHost());
                bucket.create(uri.getPath().substring(1), Files.newInputStream(temporary.toPath()));
            } else {
                Files.copy(temporary.toPath(), new File(permanent).toPath());
            }
            System.out.println("Storage.write(uri = '" + permanent + "').ok");
        } catch (Throwable t) {
            System.out.println("Storage.write(uri = '" + permanent + "').err");
            t.printStackTrace();
        }
    }
}
