gcloud config set project bk-project-90069
gcloud iam service-accounts create gsa-name \
    --description="DESCRIPTION" \
    --display-name="DISPLAY_NAME"

kubectl apply -f gke-access-gcs.ksa.yaml

gcloud iam service-accounts add-iam-policy-binding \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:cluster_project.svc.id.goog[k8s_namespace/ksa_name]" \
  gsa-name@project-90069.iam.gserviceaccount.com

