declare -r HOST="http://127.0.0.1:7080"
declare -r HEALTH_URL="${HOST}/ping"
declare -r c=5000000
declare -r runtime=jitopt
declare -r modelname=topkonly
cargo run --release --bin serving -- model_store/${modelname}_bolcom_c${c}_t50_${runtime}.pth model_store/${modelname}_bolcom_c${c}_t50_payload.yaml &
SERVING_PID=$!

while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' ${HEALTH_URL})" != "200" ]]; do
  echo waiting for server health API to return 200
  sleep 1;
done

for throttle in 1000000
do
  for qty_http_conn in 1
  do
    filename=${modelname}_c${c}_${runtime}_u${qty_http_conn}_throttle${throttle}
    cargo run --release --bin loadtest -- \
      --host ${HOST} \
      -u${qty_http_conn} \
      -t1m \
      --throttle-requests ${throttle} \
      --report-file=${filename}_report.html \
      --request-log ${filename}_log.log --request-format csv
  done
done

echo shutting down server
kill ${SERVING_PID}
