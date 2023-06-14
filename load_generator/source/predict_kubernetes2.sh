generate_post_data()
{
  cat <<EOF
{"instances":[{"context":[2]}],"parameters":[{"runtime":"eager"}]}
EOF
}

curl \
-X POST \
-H "Content-Type: application/json" \
http://34.90.244.41:7080/predictions/model/1.0 \
--data "$(generate_post_data)"



