#!/bin/bash

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
    {"runtime":  ""}
  ]
}

EOF
}

curl \
-i \
-X POST \
-H "Content-Type: application/json" \
http://127.0.0.1:8080/predictions/model/1.0/ \
--data "$(generate_post_data)"


