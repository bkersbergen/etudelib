
import com.google.api.HttpBody;
import com.google.api.core.ApiFuture;
import com.google.api.core.ApiFutures;
import com.google.api.gax.core.BackgroundResource;
import com.google.api.gax.paging.AbstractFixedSizeCollection;
import com.google.api.gax.paging.AbstractPage;
import com.google.api.gax.paging.AbstractPagedListResponse;
import com.google.api.gax.rpc.PageContext;
import com.google.api.gax.rpc.UnaryCallable;
import com.google.cloud.aiplatform.v1.*;
import com.google.cloud.aiplatform.v1.stub.PredictionServiceStub;
import com.google.cloud.aiplatform.v1.stub.PredictionServiceStubSettings;
import com.google.cloud.location.GetLocationRequest;
import com.google.cloud.location.ListLocationsRequest;
import com.google.cloud.location.ListLocationsResponse;
import com.google.cloud.location.Location;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.iam.v1.GetIamPolicyRequest;
import com.google.iam.v1.Policy;
import com.google.iam.v1.SetIamPolicyRequest;
import com.google.iam.v1.TestIamPermissionsRequest;
import com.google.iam.v1.TestIamPermissionsResponse;
import com.google.protobuf.ListValue;
import com.google.protobuf.Value;
import com.google.protobuf.util.JsonFormat;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import javax.annotation.Generated;

public class PredictionServiceClientSample  {

    @Test
    public void happyflow() throws IOException {
        String instance = "[{\"instances\": [{\"context\": [9300000080086393]}], \"parameters\": [{\"runtime\":  \"onnx\"}]}]";
        String project = "bolcom-pro-reco-analytics-fcc";
        String endpointId = "2765993023484461056";
        predictCustomTrainedModel(project, endpointId, instance);
    }

    static void predictCustomTrainedModel(String project, String endpointId, String instance)
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

            PredictRequest predictRequest =
                    PredictRequest.newBuilder()
                            .setEndpoint(endpointName.toString())
                            .addAllInstances(instanceList)
                            .build();
            System.out.println(predictRequest.toString());
            PredictResponse predictResponse = null;
            Boolean useAsync = true;
            if (useAsync) {
                ApiFuture<PredictResponse> future =
                        predictionServiceClient.predictCallable().futureCall(predictRequest);
                predictResponse = future.get();
            } else {
                predictResponse = predictionServiceClient.predict(predictRequest);
            }


            System.out.println("Predict Custom Trained model Response");
            System.out.format("\tDeployed Model Id: %s\n", predictResponse.getDeployedModelId());
            System.out.format("\tDeployed Model name: %s\n", predictResponse.getModelDisplayName());
            System.out.println("Predictions");
            for (Value prediction : predictResponse.getPredictionsList()) {
                System.out.format("\tPrediction: %s\n", prediction);
            }
        } catch (ExecutionException e) {
            throw new RuntimeException(e);

        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}