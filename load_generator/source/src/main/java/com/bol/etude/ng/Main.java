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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.bol.etude.ng.Tester.rampWithBackPressure;
import static java.time.Duration.ofMinutes;
import static java.time.Duration.ofSeconds;

public class Main {

    private static final Gson gson = new Gson();

    public static void main(String[] args) throws InterruptedException, IOException {
//        VERTEX_ENDPOINT=https://europe-west4-aiplatform.googleapis.com/v1/projects/1077776595046/locations/europe-west4/endpoints/7779079950887288832:predict;CATALOG_SIZE=1000000;REPORT_LOCATION=gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/noop_bolcom_c1000000_t50_jitopt_n1highmem4.avro;TARGET_RPS=500;RAMP_DURATION_MINUTES=5
        String endpoint_arg = System.getenv("VERTEX_ENDPOINT");
//        endpoint_arg = "https://europe-west4-aiplatform.googleapis.com/v1/projects/1077776595046/locations/europe-west4/endpoints/1677757986962931712:predict";
//        endpoint_arg = "https://httpbin.org/anything";
        System.out.println("ENV_VAR[VERTEX_ENDPOINT] = '" + endpoint_arg + "'");

        String catalog_size_arg = System.getenv("CATALOG_SIZE");
//        catalog_size_arg = "5000000";
        System.out.println("ENV_VAR[CATALOG_SIZE] = '" + catalog_size_arg + "'");

        String report_location_arg = System.getenv("REPORT_LOCATION");
//        report_location_arg = "gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/xxx.avro";
//        report_location_arg = "/tmp/etude.avro";
        System.out.println("ENV_VAR[REPORT_LOCATION] = '" + report_location_arg + "'");

        String target_rps_arg = System.getenv("TARGET_RPS");
//        target_rps_arg = 1000
        System.out.println("ENV_VAR[TARGET_RPS] = '" + target_rps_arg + "'");

        String ramp_duration_minutes_arg = System.getenv("RAMP_DURATION_MINUTES");
//        ramp_duration_minutes_arg = 20
        System.out.println("ENV_VAR[RAMP_DURATION_MINUTES] = '" + ramp_duration_minutes_arg + "'");


        if (Strings.isNullOrEmpty(endpoint_arg) ||
                Strings.isNullOrEmpty(catalog_size_arg) ||
                Strings.isNullOrEmpty(report_location_arg) ||
                Strings.isNullOrEmpty(target_rps_arg) ||
                Strings.isNullOrEmpty(ramp_duration_minutes_arg)
        ) {
            System.out.println("env variables [VERTEX_ENDPOINT, CATALOG_SIZE, RUNTIME, REPORT_LOCATION, TARGET_RPS, RAMP_DURATION_MINUTES ] are required.");
            Thread.sleep(300_000);
            System.exit(1);
        }

        System.out.println("Runtime.cores('" + Runtime.getRuntime().availableProcessors() + "')");

        try {
            System.out.println("Test.start()");

            URI endpoint = URI.create(endpoint_arg);
            File temporary = new File("/tmp/etude/report.avro");
            Journeys journeys = createSyntheticJourneys(Integer.parseInt(catalog_size_arg));
            executeTestScenario(endpoint,
                    temporary,
                    journeys,
                    Integer.parseInt(target_rps_arg),
                    ofMinutes(Integer.parseInt(ramp_duration_minutes_arg)));
            writeReportToStorage(temporary, report_location_arg);

            System.out.println("Test.ok()");
            System.exit(0);
        } catch (Throwable t) {
            t.printStackTrace();
            System.out.println("Test.err()");
            Thread.sleep(300_000);
            System.exit(1);
        }
    }

    private static Journeys createSyntheticJourneys(int size) {
        System.out.println("SyntheticJourneys.create(" + size + ")");
        SyntheticJourneySupplier journeys = new SyntheticJourneySupplier(size);
        journeys.fit(5.597568416279968, 8.0E-5, 3.650557039874508);
        return new Journeys(journeys);
    }

    private static void executeTestScenario(URI endpoint, File temporary, Journeys journeys, int targetRps, Duration ramp) {
        ExecutorService executor = Executors.newFixedThreadPool(4);

        GoogleBearerAuthenticator authenticator = null;

        String hostname = endpoint.getHost();
        if (hostname != null && hostname.endsWith(".googleapis.com")) {
            // add google bearer authentication if endpoint is not localhost
            authenticator = new GoogleBearerAuthenticator();
        }

        Requester<GoogleVertexRequest> requester = new Requester<>(endpoint, authenticator);
        Persister<Report> persister = new DataFilePersister<>(temporary, Report.class);
        Collector<Journey> collector = new Collector<>();
//        Journeys supplier = new Journeys(randomJourneySupplier());

        try (persister; requester) {
            System.out.println("Scenario.run()");

            rampWithBackPressure(targetRps, ramp, (request) -> {
                executor.execute(() -> {
                    request.fly();

                    requester.exec(journeys.pull(), (journey, success, failure) -> {
                        request.unfly();

                        Requester.Response response = success == null
                                ? new Requester.Response(Instant.EPOCH, 500, "", Duration.ofMillis(-1))
                                : success;

                        collector.add(journey, response);

                        if (!journey.last()) {
                            journeys.push(journey);
                            System.out.println(journey);
                        } else {
                            Report report = buildJourneyReport(journey, collector.remove(journey), gson);
                            persister.accept(report);
                        }
                    });

                    request.doOnTickStart(() -> {
                        try {
                            persister.flush();
                        } catch (IOException e) {
                            // ...
                        }
                    });
                });
            });

            System.out.println("Scenario.ok()");
        } catch (Exception err) {
            System.out.println("Scenario.err()");
            err.printStackTrace();
            throw new RuntimeException(err);
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
            interaction.setStatus(response.status);

            if (Strings.isNullOrEmpty(response.body)) {
                System.out.println("GoogleVertexResponse(status ='" + response.status + "').body().empty");
                applyInteractionErrorValues(interaction);
            } else {
                try {
                    System.out.println(response.body);
                    GoogleVertexResponse vertex = gson.fromJson(response.body, GoogleVertexResponse.class);
                    GoogleVertexResponse.Prediction prediction = vertex.predictions.get(0);
                    interaction.setOutput(prediction.predictions);
                    interaction.setPreprocessingMillis(prediction.timings.preprocessing);
                    interaction.setInferencingMillis(prediction.timings.inferencing);
                    interaction.setProcessingMillis(prediction.timings.postprocessing);
                } catch (Throwable t) {
                    System.out.println(response.body);
                    System.out.println("GoogleVertexResponse.parse().err + " + t);
                    applyInteractionErrorValues(interaction);
                }
            }

            interactions.add(interaction.build());
        }

        Report.Builder report = Report.newBuilder();
        report.setInteractions(interactions);
        return report.build();
    }

    private static void applyInteractionErrorValues(Interaction.Builder interaction) {
        interaction.setStatus(500);
        interaction.setOutput(Collections.emptyList());
        interaction.setPreprocessingMillis(-1);
        interaction.setInferencingMillis(-1);
        interaction.setProcessingMillis(-1);
    }

    private static void writeReportToStorage(File temporary, String permanent) throws IOException {
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
        } catch (Throwable err) {
            System.out.println("Storage.write(uri = '" + permanent + "').err");
            throw err;
        }
    }
}
