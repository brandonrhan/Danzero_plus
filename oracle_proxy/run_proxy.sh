#!/bin/bash
# Script to run the Oracle Proxy Server

# Default values
PROXY_PORT=${PROXY_PORT:-23457}
UPSTREAM_HOST=${UPSTREAM_HOST:-localhost}
UPSTREAM_PORT=${UPSTREAM_PORT:-23456}

echo "=== Oracle Proxy Server ==="
echo "Proxy listening on port: $PROXY_PORT"
echo "Forwarding to: $UPSTREAM_HOST:$UPSTREAM_PORT"
echo ""

cd "$(dirname "$0")"

python3 proxy_server.py \
    --proxy-port "$PROXY_PORT" \
    --upstream-host "$UPSTREAM_HOST" \
    --upstream-port "$UPSTREAM_PORT" \
    "$@"
