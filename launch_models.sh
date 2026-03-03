#!/bin/bash

# Start Xinference in the background
xinference-local --host 0.0.0.0 --port 9997 &

# Wait for Xinference to be ready
echo "Waiting for Xinference to start..."
until curl -s http://localhost:9997/v1/status > /dev/null; do
  sleep 2
done
echo "Xinference is ready."

# Register models
echo "Registering ruri-v3-310m..."
xinference register --model-type embedding --file /data/ruri_v3_310m.json --persist

echo "Registering ruri-v3-reranker-large..."
xinference register --model-type rerank --file /data/ruri_v3_reranker_large.json --persist

# Launch models
echo "Launching ruri-v3-310m..."
xinference launch --model-name ruri-v3-310m --model-type embedding

echo "Launching ruri-v3-reranker-large..."
xinference launch --model-name ruri-v3-reranker-large --model-type rerank

# Keep the container running by waiting for the background process
wait
