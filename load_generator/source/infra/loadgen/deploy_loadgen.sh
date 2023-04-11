kubectl --context bolcom-pro-default --namespace reco-analytics delete job loadgen --ignore-not-found=true --timeout=10m
kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f ./deploy_loadgen_job.yaml
# reco-analytics
