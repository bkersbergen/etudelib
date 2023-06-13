Use wizard to create a project and copy-paste the commandline statement

manually enable storage, container etc api's



# Set as default project and fetch kubernetes cluster endpoint and auth data.
`
gcloud config set project bk47475

gcloud container clusters get-credentials autopilot-cluster-2 \
    --region europe-west4 \
    --project=bk47475
`
`
Fetching cluster endpoint and auth data.
kubeconfig entry generated for autopilot-cluster-2.
`

AIM
take existing service account for project in AIM like 626522691289-compute@developer.gserviceaccount.com

edit the role for that account and add role 'Storage Admin'



