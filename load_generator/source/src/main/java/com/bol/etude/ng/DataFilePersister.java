package com.bol.etude.ng;

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileWriter;
import org.apache.avro.specific.SpecificDatumWriter;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.concurrent.ConcurrentLinkedQueue;

public class DataFilePersister<T> implements Persister<T> {
    private final DataFileWriter<T> writer;

    private final ConcurrentLinkedQueue<T> backlog = new ConcurrentLinkedQueue<>();

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
        backlog.add(obj);
    }

    @Override
    public void flush() {
        try {
            int size = backlog.size();
            for(int i = 0; i < size; i++) {
                var obj = backlog.poll();
                writer.append(obj);
            }
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() throws IOException {
        flush();
        writer.close();
    }
}
