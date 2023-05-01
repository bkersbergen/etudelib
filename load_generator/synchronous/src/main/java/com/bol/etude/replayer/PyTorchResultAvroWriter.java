package com.bol.etude.replayer;

import com.bol.etude.generated.PyTorchResult;
import org.apache.avro.file.DataFileWriter;
import org.apache.avro.specific.SpecificDatumWriter;

import java.io.File;
import java.io.IOException;

public class PyTorchResultAvroWriter {

    private final DataFileWriter<PyTorchResult> outputWriter;

    public PyTorchResultAvroWriter(File outputfile) throws IOException {

        DataFileWriter<PyTorchResult> dataFileWriter = new DataFileWriter<>(new SpecificDatumWriter<>(PyTorchResult.class));
        this.outputWriter = dataFileWriter.create(PyTorchResult.SCHEMA$, outputfile);
    }

    public void append(PyTorchResult pyTorchResult) throws IOException {
        this.outputWriter.append(pyTorchResult);
    }

    public void close() throws IOException {
        this.outputWriter.close();
    }
}
