PROJECT_ID="bolcom-pro-reco-analytics-fcc"

generate_post_data()
{
  cat <<EOF
{
  "instances": [
    {
  "context": [
   1, 2, 3
  ]
}
  ],
  "parameters": [
    {"runtime":  "$RUNTIME"}
  ]
}

EOF
}

curl -H "Content-Type: application/json" -X POST http://localhost:8080/predictions/model/1.0/ -d "$(generate_post_data)"
