generate_post_data()
{
  cat <<EOF
{
  "instances": [
    {
  "context": [
   1,2,3
  ]
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
-H "Content-Type: application/json" \
http://localhost:8080/predictions/model/  \
--data "$(generate_post_data)"
