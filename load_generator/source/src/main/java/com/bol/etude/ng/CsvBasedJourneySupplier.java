package com.bol.etude.ng;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import  java.util.regex.Matcher;

public class CsvBasedJourneySupplier implements Supplier<List<Long>> {
    private int C = 0;
    private List<String> sessions;
    private AtomicInteger idx = new AtomicInteger(0);
    private final Lock lock = new ReentrantLock();
    Pattern pattern = Pattern.compile("\\[(.*?)\\]");
    public CsvBasedJourneySupplier(int C, String csvFilename) {
        System.out.println(this.getClass().getSimpleName() + "( " + C + ", " + csvFilename + " )");
        this.C = C;
        this.sessions = readTextFileFromResources(csvFilename);
    }

    @Override
    public List<Long> get() {
        int currentIdx = idx.getAndIncrement();
        if (currentIdx >= sessions.size()) {
            lock.lock();
            try {
                currentIdx = idx.get();
                // Double-check within the locked section to prevent multiple threads from resetting idx
                if (currentIdx >= sessions.size()) {
                    idx.set(0);
                    currentIdx = 0;
                }
            } finally {
                lock.unlock();
            }
        }
        String unparsedSession = this.sessions.get(currentIdx);
        List<Long> itemsFromOneSession = parseLongsFromString(unparsedSession);
        for (int i = 0; i < itemsFromOneSession.size(); i++) {
            long currentValue = itemsFromOneSession.get(i);
            if (currentValue > C) {
                itemsFromOneSession.set(i, currentValue % C);
            }
        }

        return itemsFromOneSession;
    }


    static List<String> readTextFileFromResources(String fileName) {
        List<String> lines = new ArrayList<>();

        // Get the class loader
        ClassLoader classLoader = CsvBasedJourneySupplier.class.getClassLoader();

        // Get the input stream of the file from the resources folder
        try (InputStream inputStream = classLoader.getResourceAsStream(fileName)) {
            if (inputStream != null) {
                InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
                BufferedReader reader = new BufferedReader(inputStreamReader);

                // Skip the first line (column names)
                reader.readLine();

                // Read each line and store it in the list
                String line;
                while ((line = reader.readLine()) != null) {
                    lines.add(line);
                }

                reader.close();
            } else {
                System.err.println("File not found in resources: " + fileName);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return lines;
    }

    List<Long> parseLongsFromString(String lineOfText) {
        List<Long> result = new ArrayList<>();

        Matcher matcher = pattern.matcher(lineOfText);
        while (matcher.find()) {
            String[] tokens = matcher.group(1).split(",");
            for (String token : tokens) {
                result.add(Long.parseLong(token));
            }
        }
        return result;
    }

}
