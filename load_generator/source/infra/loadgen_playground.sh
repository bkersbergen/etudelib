DIR="$(dirname "$0")"

"${DIR}"/loadgen/deploy_loadgen.sh http://34.91.28.133:8080/predictions/model/1.0/ 1000000 gs://bk47476-shared/results/static.avro 300 10