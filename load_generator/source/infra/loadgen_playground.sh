DIR="$(dirname "$0")"
PROJECT_ID=bk47478

"${DIR}"/loadgen/deploy_loadgen.sh http://10.58.130.32:8080/predictions/model/1.0 100000 gs://${PROJECT_ID}-shared/results/static.avro 1000 10