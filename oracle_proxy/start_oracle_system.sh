#!/bin/bash
# Start the Oracle training system
# This runs: guandan server -> oracle proxy -> game_oracle clients

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORACLE_DIR="$(dirname "$SCRIPT_DIR")"
ACTOR_DIR="$ORACLE_DIR/actor_n"
PROXY_DIR="$ORACLE_DIR/oracle_proxy"

# Number of episodes (default 1000)
EPISODES=${1:-1000}

echo "=== DanZero Oracle Training System ==="
echo "Episodes: $EPISODES"
echo ""

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -f "guandan" 2>/dev/null
pkill -f "proxy_server.py" 2>/dev/null
pkill -f "game_oracle.py" 2>/dev/null
sleep 1

# Start the original guandan server on port 23456
echo "[1/3] Starting guandan server on port 23456..."
cd "$ACTOR_DIR"
./guandan $EPISODES > /tmp/guandan_server.log 2>&1 &
GUANDAN_PID=$!
echo "      PID: $GUANDAN_PID"
sleep 2

# Check if server started
if ! kill -0 $GUANDAN_PID 2>/dev/null; then
    echo "ERROR: guandan server failed to start"
    cat /tmp/guandan_server.log
    exit 1
fi

# Start the Oracle Proxy on port 23457
echo "[2/3] Starting Oracle Proxy on port 23457..."
cd "$PROXY_DIR"
python3 proxy_server.py --proxy-port 23457 --upstream-port 23456 > /tmp/oracle_proxy.log 2>&1 &
PROXY_PID=$!
echo "      PID: $PROXY_PID"
sleep 2

# Check if proxy started
if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo "ERROR: Oracle Proxy failed to start"
    cat /tmp/oracle_proxy.log
    kill $GUANDAN_PID 2>/dev/null
    exit 1
fi

echo "[3/3] Starting game_oracle clients (connecting to proxy port 23457)..."
cd "$ACTOR_DIR"
python3 game_oracle.py --server_port 23457 --oracle > /tmp/game_oracle.log 2>&1 &
GAME_PID=$!
echo "      PID: $GAME_PID"

echo ""
echo "=== All components started ==="
echo "  - guandan server: port 23456 (PID $GUANDAN_PID)"
echo "  - Oracle Proxy:   port 23457 (PID $PROXY_PID)"
echo "  - game_oracle:    connected (PID $GAME_PID)"
echo ""
echo "Logs:"
echo "  - Server: /tmp/guandan_server.log"
echo "  - Proxy:  /tmp/oracle_proxy.log"
echo "  - Game:   /tmp/game_oracle.log"
echo ""
echo "Press Ctrl+C to stop all components..."

# Wait for game to finish or interrupt
trap "echo 'Stopping...'; kill $GAME_PID $PROXY_PID $GUANDAN_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait $GAME_PID 2>/dev/null

# Cleanup
echo ""
echo "Game finished. Cleaning up..."
kill $PROXY_PID $GUANDAN_PID 2>/dev/null

echo "Done."
