if [ $# -ne 2 ]; then
   echo -e "\nPlease call '$0 <endpoint_id> <runtime>' to run this command!\n"
   echo -e "runtime: eager, jit or onnx"
   echo -e "4775442882221834240 eager"
   exit 1
fi

ENDPOINT_ID="$1"
RUNTIME="$2"
PROJECT_ID="bolcom-pro-reco-analytics-fcc"

generate_post_data()
{
  cat <<EOF
{
  "instances": [
    {
  "context": [
   9300000080086393
  ]
}
  ],
  "parameters": [
    {"runtime":  "$RUNTIME"}
  ]
}

EOF
}


curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://europe-west4-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/europe-west4/endpoints/${ENDPOINT_ID}:predict \
--data "$(generate_post_data)"