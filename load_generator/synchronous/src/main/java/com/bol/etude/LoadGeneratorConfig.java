package com.bol.etude;

import java.util.HashMap;
import java.util.Map;

public class LoadGeneratorConfig {
    public String connectUri = "";
    public Map<String, String> httpHeader = new HashMap<>();
    public String runtime = "onnx";
    public int qtyHttpConnections = 12;
    public String datasetName;
    public String logsPath;

    @Override
    public String toString() {
        return "LoadGeneratorConfig{" +
                "connectUri='" + connectUri + '\'' +
                ", httpHeader=" + httpHeader +
                ", runtime='" + runtime + '\'' +
                ", qtyHttpConnections=" + qtyHttpConnections +
                ", datasetName='" + datasetName + '\'' +
                ", logsPath=" + logsPath +
                '}';
    }
}
