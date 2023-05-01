package com.bol.etude;

import com.bol.etude.dataproducer.RecboleInterDataProducer;
import com.bol.etude.dataproducer.SytheticDataProducer;
import com.bol.etude.generator.LoadGenerator;
import com.bol.etude.stoppingcondition.RateLimitConfig;
import com.google.auth.oauth2.GoogleCredentials;
import org.apache.commons.cli.*;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.time.Duration;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Main {
    private static final Logger LOGGER = LogManager.getLogger(Main.class);

    public static void main(String[] args) throws InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException, IOException {
        Configurator.setLevel(Main.class.getPackageName(), Level.DEBUG);
        // command line args: "-project" "bolcom-pro-reco-analytics-fcc" "-endpointId" "4775442882221834240" "-datasetLocation" "../dataset" "-datasetName" "bolcom25m_sample"

        Option option0 = new Option("project", true, "bolcom-pro-reco-analytics-fcc");
//        Option option1 = new Option("endpointId", true, "4775442882221834240");
        Option option1 = new Option("endpointId", true, "1973306712509120512");
        Option option2 = new Option("datasetLocation", true, "../dataset");
        Option option3 = new Option("datasetName", true, "bolcom100k");

        CommandLine commandLine = process(args, option0, option1, option2, option3);

        String project = commandLine.getOptionValue("project");
        String endpointId = commandLine.getOptionValue("endpointId");
        int targetPredictionsPerSecond = 500;
        int qtyHttpConnections = 4;
        String runtime = "eager";
        String datasetLocation = commandLine.getOptionValue("datasetLocation");
        String datasetName = commandLine.getOptionValue("datasetName");
        String trainingDataPath = String.format("%s/%s/%s.inter", datasetLocation, datasetName, datasetName);
        String testDataPath = String.format("%s/%s/%s.inter", datasetLocation, datasetName, datasetName);

        GoogleCredentials sourceCredentials = GoogleCredentials.getApplicationDefault();
        if (sourceCredentials.getAccessToken() == null) {
            sourceCredentials.refresh();
        }
        String token = sourceCredentials.getAccessToken().getTokenValue();

        Map<String, String> httpHeader = new HashMap<>();
        httpHeader.put("Content-type", "application/json");
        httpHeader.put("Authorization", "Bearer " + token);

        LoadGeneratorConfig config = new LoadGeneratorConfig();
        config.connectUri = String.format("https://europe-west4-aiplatform.googleapis.com/v1/projects/%s/locations/europe-west4/endpoints/%s:predict",project, endpointId);
//        config.connectUri = String.format("http://127.0.0.1:7080/predictions/noop");
        config.httpHeader = httpHeader;
        config.qtyHttpConnections = qtyHttpConnections;
        config.datasetName = datasetName;
        config.runtime = runtime;
        config.logsPath = "logsdir";

        LOGGER.info(config);

        int minRPS = 1;

        // latency p99 for vertex endpoint is stable after 6 minutes.
        // latency p99 for torchserve is stable after 3 minutes.
        Duration duration = Duration.ofMillis((int) ((Math.sqrt(targetPredictionsPerSecond) + 30) * 1000));
        LoadGenerator loadGenerator = new LoadGenerator.Builder()
                .withConfig(config)
                .withWarmup(SytheticDataProducer.class, trainingDataPath, duration, new RateLimitConfig(minRPS, targetPredictionsPerSecond))
                .withExperiment(SytheticDataProducer.class, testDataPath)
                .withCooldown(SytheticDataProducer.class, trainingDataPath, Duration.ofSeconds(1), new RateLimitConfig(targetPredictionsPerSecond, minRPS))
                .build();
        loadGenerator.run();
    }

    public static CommandLine process(String[] args, Option... option) {
        CommandLineParser commandLineParser = new DefaultParser();
        Options options = new Options();
        Arrays.stream(option)
                .forEach(options::addOption);
        try {
            return commandLineParser.parse(options, args);
        } catch (ParseException e) {
            e.printStackTrace();
            throw new RuntimeException("Invalid arguments");
        }
    }


}