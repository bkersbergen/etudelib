package com.bol.etude.ng;

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileWriter;
import org.apache.avro.specific.SpecificDatumWriter;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;

public class DataFilePersister<T> implements Persister<T> {
    private final DataFileWriter<T> writer;

    private final ConcurrentLinkedQueue<T> backlog = new ConcurrentLinkedQueue<>();
    private AtomicBoolean writerOpen;

    DataFilePersister(@Nonnull File target, Class<T> klass) {
        try {
            Files.createDirectories(target.getAbsoluteFile().toPath().getParent());
            Schema schema$ = (Schema) klass.getDeclaredField("SCHEMA$").get(null);
            writer = new DataFileWriter<>(new SpecificDatumWriter<>(klass))
                    .create(schema$, target);
            writerOpen = new AtomicBoolean(true);
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
            for (int i = 0; i < size; i++) {
                var obj = backlog.poll();
                if (writerOpen.get()) {
                    writer.append(obj);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("DataFilePersister.flush().err");
        }
    }

    @Override
    public void close() throws IOException {
        flush();
        writerOpen.set(false);
        writer.close();
    }
}
