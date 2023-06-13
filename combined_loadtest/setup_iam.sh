#gcloud config set project bk-project-90069
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: etudelib
EOF
#
#kubectl apply -f gke-access-gcs.ksa.yaml
#
#
