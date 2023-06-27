DIR="$(dirname "$0")"
PROJECT_ID=bk47477

"${DIR}"/loadgen/deploy_loadgen.sh http://10.58.128.110:8080/predictions/model/1.0 1000000 gs://bk47476-shared/results/static.avro 1000 10