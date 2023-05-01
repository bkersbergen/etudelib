package com.bol.etude.torchserve;

public class TorchServeModelInfo {
    private String servingModelName;
    private String servingModelVersion;

    private final String modelName;
    private final String datasetName;

    private String serverIp;
    private String serverPort;

    public TorchServeModelInfo(String servingModelName, String servingModelVersion, String modelName, String datasetName, String serverIp, String serverPort) {
        this.servingModelName = servingModelName;
        this.servingModelVersion = servingModelVersion;
        this.modelName = modelName;
        this.datasetName = datasetName;
        this.serverIp = serverIp;
        this.serverPort = serverPort;
    }


    public String getServingModelName() {
        return servingModelName;
    }

    public String getServingModelVersion() {
        return servingModelVersion;
    }

    public String getModelName() {
        return modelName;
    }

    public String getDatasetName() {
        return datasetName;
    }

    public String getServerIp() {
        return serverIp;
    }

    public String getServerPort() {
        return serverPort;
    }
}
