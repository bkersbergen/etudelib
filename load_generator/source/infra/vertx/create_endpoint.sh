kubectl --context bolcom-pro-default --namespace reco-analytics delete job  vertex-create-endpoint --ignore-not-found=true --timeout=10m
kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f ./create_endpoint_job.yaml
# reco-analytics
