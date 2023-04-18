#!/usr/bin/env bash
set -e

if [ $# -lt 3 ]; then
    echo "requires args 'VERTEX_ENDPOINT', 'CATALOG_SIZE', 'REPORT_URI'"
    exit 1
fi

DIR="$(dirname "$0")"

VERTEX_ENDPOINT="${1}"
CATALOG_SIZE="${2}"
REPORT_URI="${3}"

echo "loadtest.run(endpoint = '${VERTEX_ENDPOINT}', catalog_size = '${CATALOG_SIZE}', report = '${REPORT_URI}')"

HASH=$(sum <<< "${VERTEX_ENDPOINT}" | cut -f 1 -d ' ')
JOB_NAME="etude-run-loadtest-${HASH}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --grace-period=0 --wait=true --ignore-not-found=true --timeout=5m

export JOB_NAME VERTEX_ENDPOINT CATALOG_SIZE REPORT_URI
envsubst < "$DIR"/deploy_loadgen_job.yaml > "/tmp/deploy_loadgen_job.${HASH}.yaml"
export -n JOB_NAME VERTEX_ENDPOINT CATALOG_SIZE REPORT_URI

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/deploy_loadgen_job.${HASH}.yaml"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=30m)

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow)
[[ "$LOGS" =~ .*"killing loadgen".* ]] && {
  echo "$LOGS"
  echo "loadtest.done().err"
  exit 1
}

echo "loadtest.done().ok"
exit 0


