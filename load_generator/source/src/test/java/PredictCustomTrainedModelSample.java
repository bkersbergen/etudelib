
import com.google.cloud.aiplatform.v1.EndpointName;
import com.google.cloud.aiplatform.v1.PredictRequest;
import com.google.cloud.aiplatform.v1.PredictResponse;
import com.google.cloud.aiplatform.v1.PredictionServiceClient;
import com.google.cloud.aiplatform.v1.PredictionServiceSettings;
import com.google.protobuf.ListValue;
import com.google.protobuf.Value;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.util.List;

public class PredictCustomTrainedModelSample {
    public static void main(String[] args) throws IOException {
        String instance = "[{\"context\": [9300000080086393 ]}]";
        String parameters = "[{\"runtime\":  \"onnx\"}]";
        String project = "bolcom-pro-reco-analytics-fcc";
        String endpointId = "2765993023484461056";
        predictCustomTrainedModel(project, endpointId, instance, parameters);
    }

    static void predictCustomTrainedModel(String project, String endpointId, String instance, String parameters)
            throws IOException {
        PredictionServiceSettings predictionServiceSettings =
                PredictionServiceSettings.newBuilder()
                        .setEndpoint("europe-west4-aiplatform.googleapis.com:443")
                        .build();

        // Initialize client that will be used to send requests. This client only needs to be created
        // once, and can be reused for multiple requests. After completing all of your requests, call
        // the "close" method on the client to safely clean up any remaining background resources.
        try (PredictionServiceClient predictionServiceClient =
                     PredictionServiceClient.create(predictionServiceSettings)) {
            String location = "europe-west4";
            EndpointName endpointName = EndpointName.of(project, location, endpointId);

            ListValue.Builder listValue = ListValue.newBuilder();
            JsonFormat.parser().merge(instance, listValue);
            List<Value> instanceList = listValue.getValuesList();
            ListValue.Builder parameterValue = ListValue.newBuilder();
            JsonFormat.parser().merge(parameters, parameterValue);

            PredictRequest predictRequest =
                    PredictRequest.newBuilder()
                            .setEndpoint(endpointName.toString())
                            .addAllInstances(instanceList)
                            .setParameters(parameterValue.getValuesList().get(0))
                            .build();
            System.out.println(predictRequest);
            PredictResponse predictResponse = predictionServiceClient.predict(predictRequest);

            System.out.println("Predict Custom Trained model Response");
            System.out.format("\tDeployed Model Id: %s\n", predictResponse.getDeployedModelId());
            System.out.format("\tDeployed Model name: %s\n", predictResponse.getModelDisplayName());
            System.out.println("Predictions");
            for (Value prediction : predictResponse.getPredictionsList()) {
                System.out.format("\tPrediction: %s\n", prediction);
            }
        }
    }
}