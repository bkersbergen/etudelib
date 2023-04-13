if [ $# -ne 1 ]; then
   echo -e "\nPlease call '$0 <modelname>' to run this command!\n"
   echo -e "\ne.g. '$0 lightsans_bolcom25m onnx'"
   exit 1
fi

MODELNAME="$1"

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
http://localhost:8080/predictions/${MODELNAME}/  \
--data "$(generate_post_data)"
