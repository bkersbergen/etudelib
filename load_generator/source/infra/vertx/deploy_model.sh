kubectl --context bolcom-pro-default --namespace reco-analytics delete job vertex-deploy-model --ignore-not-found=true --timeout=10m
kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f ./deploy_model_job.yaml
