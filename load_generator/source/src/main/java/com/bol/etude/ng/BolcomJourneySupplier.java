package com.bol.etude.ng;

import java.io.*;
import java.util.*;
import java.util.function.Supplier;


public class BolcomJourneySupplier implements Supplier<List<Long>> {
    private int C = 0;
    private List<String> sessions;
    private int idx = 0;
    public BolcomJourneySupplier(int C) {
        System.out.println("new BolcomJourneySupplier()");
        this.C = C;
        this.sessions = readTextFileFromResources("raw_click_lists.csv");
    }

    @Override
    public List<Long> get() {
        String unparsedSessionOrNull = this.sessions.get(idx);
        if (unparsedSessionOrNull == null) {
//            reset the index to start over again
            this.idx = 0;
            unparsedSessionOrNull = this.sessions.get(idx);
        }
        List<Long> itemsFromOneSession = parseLongsFromString(unparsedSessionOrNull);
        return itemsFromOneSession;
    }


    static List<String> readTextFileFromResources(String fileName) {
        List<String> lines = new ArrayList<>();

        // Get the class loader
        ClassLoader classLoader = BolcomJourneySupplier.class.getClassLoader();

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

    static List<Long> parseLongsFromString(String input) {
        List<Long> result = new ArrayList<>();

        // Remove the brackets and split the string by commas
        String[] parts = input.substring(2, input.length() - 2).split(",");

        // Convert the substrings to integers and add them to the list
        for (String part : parts) {
            result.add(Long.parseLong(part.trim()));
        }

        return result;
    }

}
