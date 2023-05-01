package com.bol.etude.dataproducer;

import java.time.Instant;
import java.util.List;

public class DataframeStatistics {

    public static void printDataframeStatistics(String descriptiveName, List<Row> input_df) {
        long unique_session_ids = input_df.stream().map(row -> row.getSessionId()).distinct().count();
        long unique_item_ids = input_df.stream().map(row -> row.getItemId()).distinct().count();
        int minTime = input_df.stream().map(row -> row.getTime()).reduce(Integer.MAX_VALUE, Integer::min);
        int maxTime = input_df.stream().map(row -> row.getTime()).reduce(Integer.MIN_VALUE, Integer::max);
        long unique_interactions = input_df.stream().map(row -> row.getSessionId() + "_" + row.getItemId()).distinct().count();
        double sparsity = 100 - ((100.0 / (double)(unique_session_ids * unique_item_ids)) * unique_interactions);
        System.out.println("Dataframe statistics: " + descriptiveName);
        System.out.println("\tQty item events: " + input_df.size());
        System.out.println("\tQty unique sessions: " + unique_session_ids);
        System.out.println("\tQty unique items: " + unique_item_ids);
        System.out.format("\tSparsity of dataset: %.5f percent\n", sparsity);
        System.out.println("\tSpan: " + Instant.ofEpochSecond(minTime).toString() + " / " +
                Instant.ofEpochSecond(maxTime).toString());
    }
}