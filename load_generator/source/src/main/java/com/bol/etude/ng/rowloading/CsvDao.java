package com.bol.etude.ng.rowloading;

import com.bol.etude.ng.CsvBasedJourneySupplier;
import com.bol.etude.ng.SyntheticJourneySupplier;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CsvDao {
    private final String inputFilename;
    Pattern pattern = Pattern.compile("\\[(.*?)\\]");

    public CsvDao(String inputFilename) {
        this.inputFilename = inputFilename;
    }

    public List<SyntheticJourneySupplier.Row> readRows() {
        // Get the class loader
        ClassLoader classLoader = CsvBasedJourneySupplier.class.getClassLoader();

        List<SyntheticJourneySupplier.Row> rows = new ArrayList<>();
        // Get the input stream of the file from the resources folder
        try (InputStream inputStream = classLoader.getResourceAsStream(inputFilename)) {
            if (inputStream != null) {
                InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
                BufferedReader reader = new BufferedReader(inputStreamReader);

                // Skip the first line (column names)
                reader.readLine();

                // Read each line and store it in the list
                String line;
                int sessionId = 0;
                int time = 0;
                while ((line = reader.readLine()) != null) {
                    List<Long> items = parseLongsFromString(line);
                    for (long itemId : items) {
                        SyntheticJourneySupplier.Row row = new SyntheticJourneySupplier.Row(sessionId, itemId, time);
                        rows.add(row);
                        time +=1;
                    }
                    sessionId += 1;
                }

                reader.close();
            } else {
                System.err.println("File not found in resources: " + inputFilename);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return rows;
    }

    private List<Long> parseLongsFromString(String lineOfText) {
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
