package com.bol.etude.replayer;

import com.bol.etude.generated.PyTorchResult;
import com.bol.etude.torchserve.Utils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.ConcurrentLinkedQueue;

public class TextResultWriterTask implements ResultWriterTask {
    private ConcurrentLinkedQueue<PyTorchResult> inputs;
    private PrintWriter outputWriter = null;
    private volatile boolean keepRunning;

    private static final Logger LOGGER = LogManager.getLogger(ResultWriterTask.class);

    private static final String LOGS_DIR_LOCATION = "logsdir";

    public TextResultWriterTask(ConcurrentLinkedQueue<PyTorchResult> inputs, String friendlyModelName) throws IOException {
        this.inputs = inputs;
        keepRunning = true;

        Files.createDirectories(Paths.get(LOGS_DIR_LOCATION));
        outputWriter = new PrintWriter(new BufferedWriter(new FileWriter(LOGS_DIR_LOCATION + File.separator + friendlyModelName + ".txt")));
    }

    @Override
    public void run() {
        while (keepRunning) {
            PyTorchResult pyTorchResult = inputs.poll();
            if (pyTorchResult != null) {
                outputWriter.println(pyTorchResult);
            } else {
                Utils.sleep(10);
            }
        }
        try {
            outputWriter.close();
        } catch (Exception e) {
            LOGGER.error(e);
        }
    }

    public void stop() {
        LOGGER.debug("stop()");
        keepRunning = false;
    }
}
