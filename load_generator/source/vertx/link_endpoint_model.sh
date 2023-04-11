kubectl --context bolcom-pro-default --namespace reco-analytics delete job vertex-link-endpoint-model --ignore-not-found=true --timeout=10m
kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f ./link_endpoint_model_job.yaml
