if [ $# -ne 1 ]; then
   echo -e "\nPlease call '$0 <endpoint_id>' to run this command!\n"
   echo -e "4775442882221834240"
   exit 1
fi

ENDPOINT_ID="$1"
PROJECT_ID="bolcom-pro-reco-analytics-fcc"

generate_post_data()
{
  cat <<EOF
{
  "instances": [
    {
  "data": "$( base64 -i daisy.jpg )"
}
  ],
  "parameters": [
    {"runtime":  ""}
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



