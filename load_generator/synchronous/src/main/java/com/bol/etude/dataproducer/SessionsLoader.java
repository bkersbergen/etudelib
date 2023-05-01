package com.bol.etude.dataproducer;

import com.google.cloud.storage.Blob;
import com.google.cloud.storage.Bucket;
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageOptions;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class SessionsLoader {

    private static final Logger LOGGER = LogManager.getLogger(SessionsLoader.class);

    public static List<Row> readCsv(String fullPathToCsvFile) throws IOException {
        LOGGER.info("Reading csv: {}", fullPathToCsvFile);
        URI fileLocation = URI.create(fullPathToCsvFile);

        BufferedReader br;
        if ("gs".equals(fileLocation.getScheme())){
            br = createGoogleFilesystemReader(fileLocation);
        } else {
            br = createLocalFileReader(fileLocation);
        }
        return SessionsLoader.readerToRows(br);
    }

    public static List<UserSession> getUserSessions(String training_csv_path, boolean isEvaluationData) {
        List<Row> training_df;
        try {
            training_df = SessionsLoader.readCsv(training_csv_path);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
//        DataframeStatistics.printDataframeStatistics("training data", training_df);

        // group by session_id
        Map<Integer, List<Row>> sessionIdWithRows = training_df.stream().collect(Collectors.groupingBy(Row::getSessionId,
                Collectors.toList())
        );
        // order by time ASC
        Comparator<Row> compareTime = Comparator
                .comparing(Row::getTime)
                .thenComparing(Row::getItemId);
        for (Map.Entry<Integer, List<Row>> entry : sessionIdWithRows.entrySet()) {
            entry.getValue().sort(compareTime);
        }

        // convert to different format
        List<UserSession> userSessions = new ArrayList<>(sessionIdWithRows.size());
        for (Map.Entry<Integer, List<Row>> entry : sessionIdWithRows.entrySet()) {
            long sessionId = entry.getKey();
            List<Long> items = new ArrayList<>(entry.getValue().size());
            List<Long> times = new ArrayList<>(entry.getValue().size());
            for (Row row : entry.getValue()) {
                items.add(row.getItemId());
                times.add((long) row.getTime());
            }
            if (items.size() == 0) {
                throw new IllegalStateException("wtf: " + sessionId);
            }
            UserSession userSession = new UserSession(sessionId, items, times, isEvaluationData);
            userSessions.add(userSession);
        }
        return userSessions;
    }

    private static List<Row> readerToRows(BufferedReader br) throws IOException {
        br.readLine(); // ignore header line
        List<Row> result = br.lines().map(line -> {
            String[] parts = line.split(",");
            long itemId = Long.valueOf(parts[0]);
            int time = (int)Math.round(Double.valueOf(parts[1]));
            int sessionId = Integer.valueOf(parts[2]);
            return new Row(sessionId, itemId, time);
        }).collect(Collectors.toList());
        return result;
    }

    private static BufferedReader createGoogleFilesystemReader(URI fileLocation) throws FileNotFoundException {
        Storage storageClient = StorageOptions.getDefaultInstance().getService();
        Bucket bucket = storageClient.get(fileLocation.getHost());

        String path = fileLocation.getPath();
        while (path.startsWith("/")) {
            // URI Path starts with a '/'. The Google Storage API expects paths NOT to start with a '/'.
            path = path.substring(1);
        }
        Blob blob = bucket.get(path);
        Reader reader = new StringReader(new String(blob.getContent()));
        return new BufferedReader(reader);
    }

    private static BufferedReader createLocalFileReader(URI fileLocation) throws FileNotFoundException {
        String path = fileLocation.getPath();
        return new BufferedReader(new FileReader(path));
    }
}
