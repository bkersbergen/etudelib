package com.bol.etude.ng;

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileWriter;
import org.apache.avro.specific.SpecificDatumWriter;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class DataFilePersister<T> implements Persister<T> {
    private final DataFileWriter<T> writer;

    DataFilePersister(@Nonnull File target, Class<T> klass) {
        try {
            Files.createDirectories(target.getAbsoluteFile().toPath().getParent());
            Schema schema$ = (Schema) klass.getDeclaredField("SCHEMA$").get(null);
            writer = new DataFileWriter<>(new SpecificDatumWriter<>(klass))
                    .create(schema$, target);
        } catch (IOException | NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    @Override
    public void accept(@Nonnull T obj) {
        try {
            writer.append(obj);
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() throws IOException {
        writer.close();
    }
}
