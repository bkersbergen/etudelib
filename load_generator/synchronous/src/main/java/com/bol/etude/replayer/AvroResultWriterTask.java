package com.bol.etude.replayer;


import com.bol.etude.generated.PyTorchResult;
import com.bol.etude.torchserve.Utils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.ConcurrentLinkedQueue;

public class AvroResultWriterTask implements ResultWriterTask {
    private ConcurrentLinkedQueue<PyTorchResult> inputs;
    private PyTorchResultAvroWriter outputWriter = null;
    private volatile boolean keepRunning;

    private static final Logger LOGGER = LogManager.getLogger(AvroResultWriterTask.class);


    public AvroResultWriterTask(ConcurrentLinkedQueue<PyTorchResult> inputs, String friendlyModelName, String outputPath) throws IOException {
        this.inputs = inputs;
        keepRunning = true;

        Files.createDirectories(Paths.get(outputPath));
        outputWriter = new PyTorchResultAvroWriter(new File(outputPath + File.separator + friendlyModelName + ".avro"));
    }

    @Override
    public void run() {
        while (keepRunning) {
            PyTorchResult pyTorchResult = inputs.poll();
            if (pyTorchResult == null) {
                Utils.sleep(10);
            } else {
                try {
                    outputWriter.append(pyTorchResult);
                } catch (IOException e) {
                    LOGGER.error(e);
                }
            }
        }
        try {
            outputWriter.close();
        } catch (Exception e) {
            LOGGER.error(e);
        }
        LOGGER.info("TODO: copy logsdir to final destination");
    }

    @Override
    public void stop() {
        LOGGER.debug("stop()");
        keepRunning = false;
    }
}
