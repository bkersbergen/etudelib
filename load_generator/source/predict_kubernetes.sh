if [ $# -ne 2 ]; then
   echo -e "\nPlease call '$0 <hostname> <portnumber>' to run this command!\n"
   echo -e "34.90.244.41 7080"
   exit 1
fi

HOSTNAME="$1"
PORT="$2"

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

curl \
-X POST \
-H "Content-Type: application/json" \
http://${HOSTNAME}:${PORT}/predictions/model/1.0/ \
--data "$(generate_post_data)"



