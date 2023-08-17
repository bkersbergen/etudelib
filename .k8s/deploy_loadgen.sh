#!/usr/bin/env bash
set -e

if [ $# -ne 6 ]; then
    echo "requires args 'PROJECT_ID VERTEX_ENDPOINT', 'CATALOG_SIZE', 'REPORT_LOCATION', 'TARGET_RPS', 'RAMP_DURATION_MINUTES'"
    exit 1
fi

DIR="$(dirname "$0")"

PROJECT_ID="${1}"
VERTEX_ENDPOINT="${2}"
CATALOG_SIZE="${3}"
REPORT_LOCATION="${4}"
TARGET_RPS="${5}"
RAMP_DURATION_MINUTES="${6}"

echo "$0.run(PROJECT_ID = '${PROJECT_ID}', VERTEX_ENDPOINT = '${VERTEX_ENDPOINT}', CATALOG_SIZE = '${CATALOG_SIZE}', TARGET_RPS = '${TARGET_RPS}', RAMP_DURATION_MINUTES = '${RAMP_DURATION_MINUTES}')"

HASH=$(sum <<< "${VERTEX_ENDPOINT}" | cut -f 1 -d ' ')
JOB_NAME="etude-run-loadtest-${HASH}"

kubectl delete job "${JOB_NAME}" --grace-period=0 --wait=true --ignore-not-found=true --timeout=5m

export JOB_NAME VERTEX_ENDPOINT CATALOG_SIZE REPORT_LOCATION TARGET_RPS RAMP_DURATION_MINUTES PROJECT_ID
envsubst < "$DIR"/deploy_loadgen_job.yaml > "/tmp/deploy_loadgen_job.${HASH}.yaml"
export -n JOB_NAME VERTEX_ENDPOINT CATALOG_SIZE REPORT_LOCATION TARGET_RPS RAMP_DURATION_MINUTES PROJECT_ID

kubectl apply -f - < "/tmp/deploy_loadgen_job.${HASH}.yaml"
POD_NAME=$(kubectl get pods -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl wait --for=condition=Ready pod/"$POD_NAME" --timeout=30m)

LOGS=$(kubectl logs pod/"${POD_NAME}" --follow)
if [[ "$LOGS" =~ .*"Test.ok()".* ]]; then
  echo "loadtest.ok()"
  exit 1
fi

echo "$LOGS"
echo "loadtest.err()"
exit 0


