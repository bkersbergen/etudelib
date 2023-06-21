declare -r HOST="http://127.0.0.1:7080"
declare -r HEALTH_URL="${HOST}/ping"
declare -r c=1000000
declare -r runtime=jitopt
declare -r modelname=noop
declare -r output_dir=projects
cargo run --release --bin serving -- model_store/${modelname}_bolcom_c${c}_t50_${runtime}.pth model_store/${modelname}_bolcom_c${c}_t50_payload.yaml &
SERVING_PID=$!

mkdir -p ${output_dir}

echo -n "Waiting for http server to start "
while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' ${HEALTH_URL})" != "200" ]]; do
  echo -n .
  sleep 1;
done

#for throttle in 10 50 100 250 500 1000
for throttle in 1000
do
  for qty_http_conn in 10
  do
    filename=${modelname}_c${c}_${runtime}_u${qty_http_conn}_throttle${throttle}
    cargo run --release --bin loadtest -- \
      --host ${HOST} \
      -u${qty_http_conn} \
      -t30s \
      -r10 \
      --throttle-requests ${throttle} \
      --report-file ${output_dir}/${filename}_report.html \
      --request-log ${output_dir}/${filename}_log.log --request-format csv
  done
done

echo shutting down server
kill ${SERVING_PID}
