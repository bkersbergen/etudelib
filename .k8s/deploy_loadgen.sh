#!/usr/bin/env bash
set -e

if [ $# -ne 7 ]; then
    echo "requires args 'PROJECT_ID VERTEX_ENDPOINT', 'CATALOG_SIZE', 'REPORT_LOCATION', 'TARGET_RPS', 'RAMP_DURATION_MINUTES', 'JOURNEY_SOURCE'"
    exit 1
fi

DIR="$(dirname "$0")"

PROJECT_ID="${1}"
VERTEX_ENDPOINT="${2}"
CATALOG_SIZE="${3}"
REPORT_LOCATION="${4}"
TARGET_RPS="${5}"
RAMP_DURATION_MINUTES="${6}"
JOURNEY_SOURCE="${7}"

echo "$0.run(PROJECT_ID = '${PROJECT_ID}', VERTEX_ENDPOINT = '${VERTEX_ENDPOINT}', CATALOG_SIZE = '${CATALOG_SIZE}', TARGET_RPS = '${TARGET_RPS}', RAMP_DURATION_MINUTES = '${RAMP_DURATION_MINUTES}', JOURNEY_SOURCE ='${JOURNEY_SOURCE}')"

sanitized_basename=$(basename "${REPORT_LOCATION}" | rev | cut -d. -f2- | rev | tr -cd '[:alnum:]' | cut -c 1-45)
HASH=$(echo -n "${REPORT_LOCATION}" | shasum | awk '{print $1}'| tr -cd '[:alnum:]')
JOB_NAME="${sanitized_basename}-ldg-${HASH}"
JOB_NAME=$(echo "${JOB_NAME}" | tr -cd '[:alnum:]' | cut -c 1-52)

kubectl delete job "${JOB_NAME}" --grace-period=0 --wait=true --ignore-not-found=true --timeout=5m

export JOB_NAME VERTEX_ENDPOINT CATALOG_SIZE REPORT_LOCATION TARGET_RPS RAMP_DURATION_MINUTES PROJECT_ID JOURNEY_SOURCE
envsubst < "$DIR"/deploy_loadgen_job.yaml > "/tmp/deploy_loadgen_job.${HASH}.yaml"
export -n JOB_NAME VERTEX_ENDPOINT CATALOG_SIZE REPORT_LOCATION TARGET_RPS RAMP_DURATION_MINUTES PROJECT_ID JOURNEY_SOURCE

kubectl apply -f - < "/tmp/deploy_loadgen_job.${HASH}.yaml"
POD_NAME=$(kubectl get pods -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl wait --for=condition=Ready pod/"$POD_NAME" --timeout=30m)

# Function to check the status of the Job
check_job_status() {
  local job_status
  job_status=$(kubectl get job "$JOB_NAME" -o=jsonpath='{.status.conditions[?(@.type=="Complete")].status}')

  if [ -n "$job_status" ]; then
    if [[ "$job_status" == "True" ]]; then
      echo "Job $JOB_NAME succeeded."
      exit 0
    elif [[ "$job_status" == "False" ]]; then
      echo "Job $JOB_NAME failed."
      exit 1
    else
      echo -n "."
    fi
  else
    echo -n "."
  fi
}

# Wait for the Job to complete (successfully or with failure)
while true; do
    check_job_status
    sleep 20  # Adjust the polling interval as needed
done

#
#LOGS=$(kubectl logs pod/"${POD_NAME}" --follow)
#if [[ "$LOGS" =~ .*"Test.ok()".* ]]; then
#  echo "loadtest.ok()"
#  exit 0
#fi
#
#echo "$LOGS"
#echo "loadtest.err()"
#exit 1


