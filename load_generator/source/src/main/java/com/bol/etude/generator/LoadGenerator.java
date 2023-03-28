package com.bol.etude.generator;

import com.bol.etude.LoadGeneratorConfig;
import com.bol.etude.dataproducer.RecboleInterDataProducer;
import com.bol.etude.dataproducer.UserSession;
import com.bol.etude.generated.PyTorchResult;
import com.bol.etude.replayer.AvroResultWriterTask;
import com.bol.etude.replayer.SessionsReplayerTask;
import com.bol.etude.stoppingcondition.RateLimitConfig;
import com.bol.etude.torchserve.RateLimiterFactory;
import com.bol.etude.torchserve.Utils;

import com.google.common.util.concurrent.RateLimiter;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.time.Duration;
import java.util.concurrent.*;

import static java.lang.System.exit;

public class LoadGenerator {

    private final LoadGeneratorConfig config;

    private BlockingQueue<UserSession> sessionsQueue;

    // WARMUP DATA PRODUCER
    private RecboleInterDataProducer warmupDataProducerTask;
    private Duration warmupDuration;
    private RateLimitConfig warmupRatelimitConfig;

    private final ExecutorService warmupExecutorService = Executors.newFixedThreadPool(1);

    // EXPERIMENT DATA PRODUCER
    private RecboleInterDataProducer experimentDataProducerTask;
    private final ExecutorService experimentExecutorService = Executors.newFixedThreadPool(1);

    // COOLDOWN DATA PRODUCER
    private RecboleInterDataProducer cooldownDataProducerTask;
    private Duration cooldownDuration;
    private RateLimitConfig cooldownRatelimitConfig;

    private final ExecutorService cooldownExecutorService = Executors.newFixedThreadPool(1);

    // SESSION REPLAYERS
    private ExecutorService replayerExecutorService;

    // RESULT WRITER
    private final ExecutorService resultWriterExecutorService = Executors.newFixedThreadPool(1);

    private final ConcurrentLinkedQueue<PyTorchResult> pyTorchResultQueue = new ConcurrentLinkedQueue<>();

    private static final Logger LOGGER = LogManager.getLogger(LoadGenerator.class);


    public LoadGenerator(LoadGeneratorConfig config, BlockingQueue<UserSession>sessionsQueue) throws IOException {
        this.config = config;
        this.sessionsQueue = sessionsQueue;
    }

    private void setWarmup(RecboleInterDataProducer warmupDataProducerTask, Duration duration, RateLimitConfig warmupRatelimit) {
        this.warmupDataProducerTask = warmupDataProducerTask;
        this.warmupDuration = duration;
        this.warmupRatelimitConfig = warmupRatelimit;
    }

    private void setExperiment(RecboleInterDataProducer experimentDataProducerTask) {
        this.experimentDataProducerTask = experimentDataProducerTask;
    }

    private void setCooldown(RecboleInterDataProducer cooldownDataProducerTask, Duration duration, RateLimitConfig cooldownRatelimit) {
        this.cooldownDataProducerTask = cooldownDataProducerTask;
        this.cooldownDuration = duration;
        this.cooldownRatelimitConfig = cooldownRatelimit;
    }

    private void setupFlow() throws IOException {
        int qtyServerWorkersForModel = this.config.qtyHttpConnections;
        CloseableHttpClient httpClient = Utils.createHttpClient(qtyServerWorkersForModel);
        RateLimiterFactory rlf = new RateLimiterFactory();
        final RateLimiter rateLimiter = rlf.createRateLimiter(this.warmupRatelimitConfig.getStartingValue(), this.warmupRatelimitConfig.getTargetValue(), this.warmupDuration);
        int qtySessionReplayerThreads = qtyServerWorkersForModel + 5;
        this.replayerExecutorService = Executors.newFixedThreadPool(qtySessionReplayerThreads);
        for (int i = 0; i < qtySessionReplayerThreads; i++) {
            this.replayerExecutorService.execute(new SessionsReplayerTask(sessionsQueue, pyTorchResultQueue, httpClient, config, rateLimiter));
        }
        this.resultWriterExecutorService.execute(new AvroResultWriterTask(pyTorchResultQueue, config.datasetName, config.logsPath));
        // start control the production of data.

        this.warmupExecutorService.execute(warmupDataProducerTask);
        while (!warmupDataProducerTask.isRunIsFinished()){
            Utils.sleep(250);
        }
        LOGGER.info("Warmup is finished");
        warmupExecutorService.shutdown();
        this.experimentExecutorService.execute(experimentDataProducerTask);
        while (!experimentDataProducerTask.isRunIsFinished()){
            Utils.sleep(250);
        }
        this.experimentExecutorService.shutdown();
        LOGGER.info("Main Experiment is finished");
        this.cooldownExecutorService.execute(cooldownDataProducerTask);
        while (!cooldownDataProducerTask.isRunIsFinished()){
            Utils.sleep(250);
        }
        LOGGER.info("Cooldown is finished");
        this.cooldownExecutorService.shutdown();
        Utils.sleep(1000);
        replayerExecutorService.shutdown();
        resultWriterExecutorService.shutdown();
        exit(0);
    }


    public void run() throws IOException {
        LOGGER.info("LoadGenerator.run() called");
        this.setupFlow();
    }


    public static class Builder {
        private BlockingQueue<UserSession> sessionsQueue = new LinkedBlockingQueue<>(1000);

        private RecboleInterDataProducer warmupDataProducerTask;
        private Duration warmupDuration;
        private RateLimitConfig warmupRatelimit;

        private RecboleInterDataProducer experimentDataProducerTask;


        private RecboleInterDataProducer cooldownDataProducerTask;
        private Duration cooldownDuration;
        private RateLimitConfig cooldownRatelimit;
        private LoadGeneratorConfig config;


        public Builder withWarmup(Class<RecboleInterDataProducer> dataProducerType, String trainingDataPath, Duration duration, RateLimitConfig rateLimitConfig) throws NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
            LOGGER.info("preparing warmup");
            this.warmupDataProducerTask = dataProducerType.getDeclaredConstructor().newInstance();
            this.warmupDataProducerTask.configure(trainingDataPath, sessionsQueue, false, duration);
            this.warmupDuration = duration;
            this.warmupRatelimit = rateLimitConfig;
            return this;  //By returning the builder each time, we can create a fluent interface.
        }

        public Builder withExperiment(Class<RecboleInterDataProducer> dataProducerType, String testDataPath) throws NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
            LOGGER.info("preparing experiment");
            this.experimentDataProducerTask = dataProducerType.getDeclaredConstructor().newInstance();
            this.experimentDataProducerTask.configure(testDataPath, sessionsQueue, true, Duration.ofHours(1));
            return this;
        }

        public Builder withCooldown(Class<RecboleInterDataProducer> dataProducerType, String trainingDataPath, Duration duration, RateLimitConfig rateLimitConfig) throws NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
            LOGGER.info("preparing cooldown");
            this.cooldownDataProducerTask = dataProducerType.getDeclaredConstructor().newInstance();
            this.cooldownDataProducerTask.configure(trainingDataPath, sessionsQueue, false, duration);
            this.cooldownDuration = duration;
            this.cooldownRatelimit = rateLimitConfig;
            return this;
        }

        public LoadGenerator build() throws IOException {
            LoadGenerator loadGenerator = new LoadGenerator(this.config, this.sessionsQueue);
            loadGenerator.setWarmup(this.warmupDataProducerTask, this.warmupDuration, this.warmupRatelimit);
            loadGenerator.setExperiment(this.experimentDataProducerTask);
            loadGenerator.setCooldown(this.cooldownDataProducerTask, this.cooldownDuration, this.cooldownRatelimit);
            return loadGenerator;
        }

        public Builder withConfig(LoadGeneratorConfig config) {
            this.config = config;
            LOGGER.info("configuration set");
            return this;
        }
    }


}
