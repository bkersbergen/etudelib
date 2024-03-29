#!/usr/bin/env bash
set -e

PROJECT_ID=bk47472
if [ $# -lt 5 ]; then
    echo "requires args 'VERTEX_ENDPOINT', 'CATALOG_SIZE', 'REPORT_LOCATION', 'TARGET_RPS', 'RAMP_DURATION_MINUTES'"
    exit 1
fi

DIR="$(dirname "$0")"

VERTEX_ENDPOINT="${1}"
CATALOG_SIZE="${2}"
REPORT_LOCATION="${3}"
TARGET_RPS="${4}"
RAMP_DURATION_MINUTES="${5}"

echo "loadtest.run(endpoint = '${VERTEX_ENDPOINT}', catalog_size = '${CATALOG_SIZE}', report_location = '${REPORT_LOCATION}', target_rps = '${TARGET_RPS}', ramp_duration_minutes = '${RAMP_DURATION_MINUTES}')"

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


