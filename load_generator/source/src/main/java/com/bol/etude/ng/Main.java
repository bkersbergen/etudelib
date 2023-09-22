package com.bol.etude.ng;

import com.bol.etude.generated.Interaction;
import com.bol.etude.generated.Meta;
import com.bol.etude.generated.Report;
import com.bol.etude.ng.Journeys.Journey;
import com.bol.etude.ng.Requester.Response;
import com.google.cloud.storage.*;
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
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.time.Duration.ofMinutes;

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

        String journeysource_arg = System.getenv("JOURNEY_SOURCE");
        System.out.println("ENV_VAR[JOURNEY_SOURCE] = '" + journeysource_arg + "'");

        String reportFileDestination = System.getenv("REPORT_LOCATION");
//        reportFileDestination = "gs://bolcom-pro-reco-analytics-fcc-shared/etude_reports/xxx.avro";
//        reportFileDestination = "/tmp/etude.avro";
        System.out.println("ENV_VAR[REPORT_LOCATION] = '" + reportFileDestination + "'");

        String target_rps_arg = System.getenv("TARGET_RPS");
//        target_rps_arg = 1000
        System.out.println("ENV_VAR[TARGET_RPS] = '" + target_rps_arg + "'");

        String ramp_duration_minutes_arg = System.getenv("RAMP_DURATION_MINUTES");
//        ramp_duration_minutes_arg = 20
        System.out.println("ENV_VAR[RAMP_DURATION_MINUTES] = '" + ramp_duration_minutes_arg + "'");

        if (Strings.isNullOrEmpty(endpoint_arg) ||
                Strings.isNullOrEmpty(catalog_size_arg) ||
                Strings.isNullOrEmpty(reportFileDestination) ||
                Strings.isNullOrEmpty(target_rps_arg) ||
                Strings.isNullOrEmpty(ramp_duration_minutes_arg) ||
                Strings.isNullOrEmpty(journeysource_arg)
        ) {
            System.out.println("env variables [VERTEX_ENDPOINT, CATALOG_SIZE, RUNTIME, REPORT_LOCATION, JOURNEY_SOURCE, TARGET_RPS, RAMP_DURATION_MINUTES ] are required.");
            Thread.sleep(300_000);
            System.exit(1);
        }

        System.out.println("Runtime.cores('" + Runtime.getRuntime().availableProcessors() + "')");

        File temporaryReportFile = new File("/tmp/etude/report.avro");
        File temporaryMetaFile = new File("/tmp/etude/meta.avro");
        String metaFileDestination = appendMetaToBaseFilename(reportFileDestination);

        try {
            System.out.println("Test.start()");

            URI endpoint = URI.create(endpoint_arg);

            Journeys journeys;
            switch (journeysource_arg) {
                case "synthetic_bolcom":
                    System.out.println("using dataset 'synthetic_bolcom'");
                    journeys = createSyntheticJourneys(Integer.parseInt(catalog_size_arg));
                    break;
                case "sample_bolcom":
                    System.out.println("using dataset 'sample_bolcom'");
                    journeys = createBolcomJourneys(Integer.parseInt(catalog_size_arg));
                    break;
                case "sample_yoochoose":
                    System.out.println("using dataset 'sample_yoochoose'");
                    journeys = createYoochooseJourneys(Integer.parseInt(catalog_size_arg));
                    break;
                default:
                    // Handle the case where 'a' doesn't match any strategy
                    throw new IllegalArgumentException("Invalid value of 'journeysource_arg'" + journeysource_arg);
            }

            executeTestScenario(endpoint,
                    temporaryReportFile,
                    temporaryMetaFile,
                    journeys,
                    Integer.parseInt(target_rps_arg),
                    ofMinutes(Integer.parseInt(ramp_duration_minutes_arg)));

            System.out.println("Test.ok()");
            copyResultsToBucket(temporaryReportFile, reportFileDestination, temporaryMetaFile, metaFileDestination);
        } catch (Throwable t) {
            //noinspection CallToPrintStackTrace
            t.printStackTrace();
            System.out.println("Test.err()");
            Thread.sleep(Duration.ofMinutes(1).toMillis());
        }
        System.exit(0);
    }

    private static String appendMetaToBaseFilename(String originalPath) {
        // Extract the base filename without the extension
        int lastSlashIndex = originalPath.lastIndexOf("/");
        int lastDotIndex = originalPath.lastIndexOf(".");

        if (lastSlashIndex >= 0 && lastDotIndex >= lastSlashIndex) {
            String baseFilename = originalPath.substring(lastSlashIndex + 1, lastDotIndex);
            String newBaseFilename = baseFilename + "_meta";

            // Replace the old base filename with the new one
            return originalPath.substring(0, lastSlashIndex + 1) + newBaseFilename + originalPath.substring(lastDotIndex);
        } else {
            // If the format of the input path is unexpected, return the original path
            return originalPath;
        }
    }

    private static void copyResultsToBucket(File reportSourceFile, String reportFileDestination, File temporaryMetaFile, String metaFileDestination) {
        System.out.println("copyResultsToBucket");
        try {
            copyFileTo(reportSourceFile, reportFileDestination);
            System.out.println("copy.ok(report)");
            copyFileTo(temporaryMetaFile, metaFileDestination);
            System.out.println("copy.ok(meta)");
        } catch (Throwable e) {
            //noinspection CallToPrintStackTrace
            e.printStackTrace();
            System.out.println("Report.write.err()");
            try {
                System.out.println("Thread.sleep(Duration.ofMinutes(5).toMillis());");
                Thread.sleep(Duration.ofMinutes(5).toMillis());
            } catch (InterruptedException ex) {
                // ignore, can't fix
            }
        }
//        System.out.println("Last line of code in addShutdownHook");
        try {
            Thread.sleep(Duration.ofSeconds(10).toMillis());
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private static Journeys createYoochooseJourneys(int size) {
        CsvBasedJourneySupplier journeys = new CsvBasedJourneySupplier(size, "yoochoose_sample.csv");
        return new Journeys(journeys);

    }

    private static Journeys createBolcomJourneys(int size) {
        CsvBasedJourneySupplier journeys = new CsvBasedJourneySupplier(size, "raw_click_lists.csv");
        return new Journeys(journeys);
    }

    private static Journeys createSyntheticJourneys(int size) {
        System.out.println("SyntheticJourneys.create(" + size + ")");
        SyntheticJourneySupplier journeys = new SyntheticJourneySupplier(size);
        journeys.fit(5.597568416279968, 8.0E-5, 3.650557039874508);
        return new Journeys(journeys);
    }

    private static void executeTestScenario(URI endpoint, File tempReportFile, File tempMetaFile, Journeys journeys, int targetRps, Duration ramp) throws IOException {
        ExecutorService executor = Executors.newFixedThreadPool(4);

        GoogleBearerAuthenticator authenticator = null;

        String hostname = endpoint.getHost();
        if (hostname != null && hostname.endsWith(".googleapis.com")) {
            // add google bearer authentication if endpoint is not localhost
            authenticator = new GoogleBearerAuthenticator();
        }

        Requester<GoogleVertexRequest> requester = new Requester<>(endpoint, authenticator);
        Persister<Report> reportPersister = new DataFilePersister<>(tempReportFile, Report.class);
        Persister<Meta> metaPersister = new DataFilePersister<>(tempMetaFile, Meta.class);

        Collector<Journey> collector = new Collector<>();
//        Journeys supplier = new Journeys(randomJourneySupplier());

        try (reportPersister; metaPersister; requester) {
            System.out.println("Scenario.run()");

            rampWithBackPressure(targetRps, ramp, metaPersister, (request) -> {
                executor.execute(() -> {
                    request.fly();

                    requester.exec(journeys.pull(), (journey, success, failure) -> {
                        request.unfly();

                        Requester.Response response = success == null
                                ? new Requester.Response(Instant.now(), 500, failure.getMessage(), Duration.ofMillis(-1), -1.0)
                                : success;

                        collector.add(journey, response);

                        if (!journey.last()) {
                            journeys.push(journey);
                        } else {
                            Report report = buildJourneyReport(journey, collector.remove(journey), gson);
                            reportPersister.accept(report);
                        }
                    });

                    request.doOnTickStart(() -> {
                        try {
                            reportPersister.flush();
                            metaPersister.flush();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    });
                });
            });
            System.out.println("Thread.sleep(Duration.ofSeconds(10).toMillis());");
            Thread.sleep(Duration.ofSeconds(10).toMillis());
        } catch (Exception err) {
            System.out.println("Scenario.err()");
            err.printStackTrace();
        }
        System.out.println("requester.close()");
        try {
            requester.close();
        } catch (Exception exception) {
            exception.printStackTrace();
        }
        System.out.println("reportPersister.close()");
        try {
            reportPersister.close();
        } catch (Exception exception) {
            exception.printStackTrace();
        }
        System.out.println("metaPersister.close()");
        try {
            metaPersister.close();
        } catch (Exception exception) {
            exception.printStackTrace();
        }
        try {
            System.out.println("Thread.sleep(Duration.ofSeconds(5).toMillis());");
            Thread.sleep(Duration.ofSeconds(5).toMillis());
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

    }

    private static Report buildJourneyReport(Journey journey, List<Response> responses, Gson gson) {
        ArrayList<Interaction> interactions = new ArrayList<>();

        for (int index = 0; index < journey.size(); index++) {
            if (index >= responses.size()) {
                System.out.println("Error: the amount of reponses does not match the amount of requests. journey.size() != responses.size()");
                continue;
            }
            Response response = responses.get(index);

            Interaction.Builder interaction = Interaction.newBuilder();
            interaction.setTimestampEpochMillis(response.start.toEpochMilli());
            interaction.setInput(journey.items().subList(0, index + 1));
            interaction.setLatencyMillis(response.latency.toMillis());
            interaction.setServersideLatencyMillis(response.serverSideLatency);
            interaction.setStatus(response.status);

            if (response.status != 200 || Strings.isNullOrEmpty(response.body)) {
//                System.out.println("GoogleVertexResponse(status ='" + response.status + "').body().err");
                applyInteractionErrorValues(interaction, response.status);
            } else {
                try {
                    GoogleVertexResponse vertex = gson.fromJson(response.body, GoogleVertexResponse.class);
                    interaction.setOutput(vertex.items.get(0));
                    interaction.setPreprocessingMillis(vertex.timings.preprocessing);
                    interaction.setInferencingMillis(vertex.timings.inferencing);
                    interaction.setProcessingMillis(vertex.timings.postprocessing);
                    interaction.setModelName(vertex.timings.model_name);
                    interaction.setModelThreadQty(vertex.timings.model_thread_qty);
                    interaction.setModelDevice(vertex.timings.model_device);
                } catch (Throwable t) {
                    System.out.println("GoogleVertexResponse.parse().err + " + t + "-------" + response.body);
                    applyInteractionErrorValues(interaction, response.status);
                }
            }

            interactions.add(interaction.build());
        }

        Report.Builder report = Report.newBuilder();
        report.setInteractions(interactions);
        return report.build();
    }

    private static void applyInteractionErrorValues(Interaction.Builder interaction, int status) {
        interaction.setStatus(status);
        interaction.setOutput(Collections.emptyList());
        interaction.setPreprocessingMillis(-1);
        interaction.setInferencingMillis(-1);
        interaction.setProcessingMillis(-1);
        interaction.setModelDevice("");
        interaction.setModelName("");
        interaction.setModelThreadQty(-1);
    }

    private static void copyFileTo(File sourceFile, String destination) throws IOException {
        try {
            System.out.println("Storage.write(uri = '" + destination + "')");
            if (destination.startsWith("gs://")) {
                Storage storage = StorageOptions.getDefaultInstance().getService();
                URI uri = URI.create(destination);
//                Bucket bucket = storage.get(uri.getHost());
//                bucket.create(uri.getPath().substring(1), Files.newInputStream(sourceFile.toPath()));
                byte[] data = Files.readAllBytes(sourceFile.toPath());
                BlobId blobId = BlobId.of(uri.getHost(), uri.getPath().substring(1));
                BlobInfo blobInfo = BlobInfo.newBuilder(blobId).build();
                storage.create(blobInfo, data);
            } else {
                Files.copy(sourceFile.toPath(), new File(destination).toPath(), REPLACE_EXISTING);
            }
            System.out.println("Storage.write(uri = '" + destination + "').ok");
        } catch (Throwable err) {
            System.out.println("Storage.write(uri = '" + destination + "').err");
            throw err;
        }
    }
}
