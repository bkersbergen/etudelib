How I setup my CICD infra:


1) Set as default project 
`
gcloud config set project bk47477
`
Manually enable Kubernetes API in the WebUI.
2) Create kubernetes cluster in project. Using WebUI with all defaults (Auto-pilot) except for region: WEST-4 (Amsterdam). The WebUI gave this commandline statement at the end:

PROJECT_ID=bk47477
`
gcloud container --project "${PROJECT_ID}" clusters create-auto "autopilot-cluster-1" --region "europe-west4" --release-channel "regular" --network "projects/${PROJECT_ID}/global/networks/default" --subnetwork "projects/${PROJECT_ID}/regions/europe-west4/subnetworks/default" --cluster-ipv4-cidr "/17" --services-ipv4-cidr "/22"
`
kubeconfig entry generated for autopilot-cluster-1.
NAME                 LOCATION      MASTER_VERSION  MASTER_IP      MACHINE_TYPE  NODE_VERSION    NUM_NODES  STATUS
autopilot-cluster-1  europe-west4  1.25.8-gke.500  34.90.174.242  e2-medium     1.25.8-gke.500  3          RUNNING


3) Fetch kubernetes cluster endpoint and auth data.
`
gcloud container clusters get-credentials autopilot-cluster-1 \
    --region europe-west4 \
    --project="${PROJECT_ID}"
`
`
Fetching cluster endpoint and auth data.
kubeconfig entry generated for autopilot-cluster-1.
`

4) Create Kubernetes service account
`
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: etudelib
EOF
`
5) Create GCP service account
gcloud iam service-accounts create etudelib \
    --description="etudelib" \
    --display-name="etudelib"

6) Storage bucket
gcloud storage buckets create gs://${PROJECT_ID}-shared

For read-write workloads: 
gcloud storage buckets add-iam-policy-binding gs://${PROJECT_ID}-shared \
    --member "serviceAccount:etudelib@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role "roles/storage.insightsCollectorService"
gcloud storage buckets add-iam-policy-binding gs://${PROJECT_ID}-shared \
    --member "serviceAccount:etudelib@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role "roles/storage.objectAdmin"


The relationship between kubernetes accounts and google cloud
https://github.com/GoogleCloudPlatform/gcs-fuse-csi-driver/blob/main/docs/usage.md
Bind the the Kubernetes Service Account with the GCP Service Account.
CLUSTER_PROJECT_ID=autopilot-cluster-1



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









==============================================================================================

5) Manually add the following roles in the WebUI AIM
'storage admin'
'kubernetes engine service agent'
'Storage Object Admin' to the service account https://docs.cloudera.com/HDPDocuments/HDP2/HDP-2.6.5/bk_cloud-data-access/content/edit-bucket-permissions.html
'Storage Legacy Bucket Owner' to the service account https://cloud.google.com/storage/docs/access-control/iam-roles
kubectl get pods
kubectl get pod/etude-run-loadtest-34367-khlp4 -o yaml
  `serviceAccount: etudelib
  serviceAccountName: etudelib
  `
7) errors:
                                                                                                           │
│   "code" : 403,                                                                                                                                           │
│   "errors" : [ {                                                                                                                                          │
│     "domain" : "global",                                                                                                                                  │
│     "message" : "Caller does not have storage.buckets.get access to the Google Cloud Storage bucket. Permission 'storage.buckets.get' denied on resource  │
│     "reason" : "forbidden"                                                                                                                                │
│   } ],                                                                                                                                                    │
│   "message" : "Caller does not have storage.buckets.get access to the Google Cloud Storage bucket. Permission 'storage.buckets.get' denied on resource (o │
│ }                                              


7) 
8) 
9) 
10) deploy POD
` 
kubectl apply -f torchserve_spec_t4.yaml
`

validate the POD and its serviceaccount using:
`
kubectl get pods/torchserve-noop-t4 -o yaml
`

7) Open terminal in POD
`
kubectl exec -it torchserve-noop-t4 bash
`










===============================================================================================================
old


AIM
take existing service account for project in AIM like 626522691289-compute@developer.gserviceaccount.com

edit the role for that account and add role 'Storage Admin'
When executing `gcloud auth list` it prints
`bk47475.svc.id.goog`
and not the service account

`
gcloud iam service-accounts create etudelib \
    --description="etudelib" \
    --display-name="etudelib"
`

