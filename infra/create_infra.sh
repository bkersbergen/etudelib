#!/bin/bash
set -e

if [[ $# -ne 1 ]]; then
    echo "Illegal number of parameters. PROJECT_ID is missing"
    echo "Usage: $0 bk47471"
    exit 2
fi

PROJECT_ID=$1
`
gcloud config set project ${PROJECT_ID}
`
gcloud services enable container.googleapis.com

gcloud container --project "${PROJECT_ID}" clusters create-auto "autopilot-cluster-1" --region "europe-west4" --release-channel "regular" --network "projects/${PROJECT_ID}/global/networks/default" --subnetwork "projects/${PROJECT_ID}/regions/europe-west4/subnetworks/default" --cluster-ipv4-cidr "/17" --services-ipv4-cidr "/22"
``
gcloud container clusters get-credentials autopilot-cluster-1 \
    --region europe-west4 \
    --project="${PROJECT_ID}"
``
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: etudelib
EOF

`
gcloud iam service-accounts create etudelib \
    --description="etudelib" \
    --display-name="etudelib"

gcloud storage buckets create gs://${PROJECT_ID}-shared
`
gcloud storage buckets add-iam-policy-binding gs://${PROJECT_ID}-shared \
    --member "serviceAccount:etudelib@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role "roles/storage.insightsCollectorService"
gcloud storage buckets add-iam-policy-binding gs://${PROJECT_ID}-shared \
    --member "serviceAccount:etudelib@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role "roles/storage.objectAdmin"
`

`
gcloud iam service-accounts add-iam-policy-binding etudelib@${PROJECT_ID}.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${PROJECT_ID}.svc.id.goog[default/etudelib]"

kubectl annotate serviceaccount etudelib \
    --namespace default \
    iam.gke.io/gcp-service-account=etudelib@${PROJECT_ID}.iam.gserviceaccount.com

kubectl create rolebinding etudelib-rolebinding \
   --clusterrole=view \
   --serviceaccount=default:etudelib \
   --namespace=default


