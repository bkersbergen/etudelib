How I setup my CICD infra:


1) Set as default project 
`
gcloud config set project bk47475
`

2) Create kubernetes cluster in project. Using WebUI with all defaults (Auto-pilot) except for region: WEST-4 (Amsterdam). The WebUI gave this commandline statement at the end:

`
gcloud container --project "bk47475" clusters create-auto "autopilot-cluster-1" --region "europe-west4" --release-channel "regular" --network "projects/bk47475/global/networks/default" --subnetwork "projects/bk47475/regions/europe-west4/subnetworks/default" --cluster-ipv4-cidr "/17" --services-ipv4-cidr "/22"
`
kubeconfig entry generated for autopilot-cluster-1.
NAME                 LOCATION      MASTER_VERSION  MASTER_IP      MACHINE_TYPE  NODE_VERSION    NUM_NODES  STATUS
autopilot-cluster-1  europe-west4  1.25.8-gke.500  34.90.174.242  e2-medium     1.25.8-gke.500  3          RUNNING


3) Fetch kubernetes cluster endpoint and auth data.
`
gcloud container clusters get-credentials autopilot-cluster-1 \
    --region europe-west4 \
    --project=bk47475
`
`
Fetching cluster endpoint and auth data.
kubeconfig entry generated for autopilot-cluster-1.
`

4) Create service account
`
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: etudelib
EOF
`
kubectl create rolebinding etudelib-rolebinding \
   --clusterrole=view \
   --serviceaccount=default:etudelib \
   --namespace=default

5) Manually add the following roles in the WebUI AIM
'storage admin'
'kubernetes engine service agent'

6) deploy POD
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

