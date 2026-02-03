#!/bin/bash
# Test script to verify the Oracle Proxy Server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORACLE_DIR="$(dirname "$SCRIPT_DIR")"
ACTOR_DIR="$ORACLE_DIR/actor_n"

echo "=== Oracle Proxy Test ==="
echo ""

# Check if guandan server exists
if [ ! -f "$ACTOR_DIR/guandan" ]; then
    echo "ERROR: guandan server not found at $ACTOR_DIR/guandan"
    exit 1
fi

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -f "guandan" 2>/dev/null
pkill -f "proxy_server.py" 2>/dev/null
sleep 1

# Start the original guandan server
echo "Starting guandan server on port 23456..."
cd "$ACTOR_DIR"
./guandan 100 &
GUANDAN_PID=$!
echo "  PID: $GUANDAN_PID"
sleep 2

# Check if server started
if ! kill -0 $GUANDAN_PID 2>/dev/null; then
    echo "ERROR: guandan server failed to start"
    exit 1
fi

# Start the proxy server
echo "Starting Oracle Proxy on port 23457..."
cd "$SCRIPT_DIR"
python3 proxy_server.py --proxy-port 23457 --upstream-port 23456 &
PROXY_PID=$!
echo "  PID: $PROXY_PID"
sleep 2

# Check if proxy started
if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo "ERROR: Proxy server failed to start"
    kill $GUANDAN_PID 2>/dev/null
    exit 1
fi

echo ""
echo "Both servers running. Starting test clients..."
echo ""

# Run the test
python3 test_proxy.py ws://localhost:23457

# Cleanup
echo ""
echo "Cleaning up..."
kill $PROXY_PID 2>/dev/null
kill $GUANDAN_PID 2>/dev/null

echo "Done."
