DIR="$(dirname "$0")"

"${DIR}"/loadgen/deploy_loadgen.sh http://34.147.33.219:8080/predictions/model/1.0/ 1000000 gs://bk47475-shared/results/noop.avro 1000 10