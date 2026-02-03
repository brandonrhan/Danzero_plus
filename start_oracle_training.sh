#!/bin/bash
# Oracle Training Startup Script for DanZero
# 
# This script starts all components for oracle training:
# 1. Guandan game server (original)
# 2. Oracle proxy server (injects all hand cards)
# 3. Oracle learner (receives training data, updates Q-network)
# 4. Oracle game clients (play games, collect data)
# 5. Oracle actors (run Q-network inference, send data to learner)
#
# Architecture:
#   [Guandan Server:23456] <-> [Oracle Proxy:23457] <-> [Game Clients] <-> [Actors:6000-6003] <-> [Learner:5000-5001]
#
# Oracle State Dimensions:
#   - Original state: 513 dims (without action)
#   - Oracle addition: 162 dims (3 players * 54 cards each)
#   - Action: 54 dims
#   - Total: 729 dims (with action), 675 dims (without action)

set -e

# Activate conda environment with TensorFlow 1.15
source ~/miniconda/etc/profile.d/conda.sh
conda activate tf115
echo "Using Python: $(which python)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Ports
GAME_SERVER_PORT=23456
ORACLE_PROXY_PORT=23457
LEARNER_DATA_PORT=5000
LEARNER_PARAM_PORT=5001

# Paths
GUANDAN_SERVER="$PROJECT_ROOT/guandan_server"
ORACLE_PROXY="$PROJECT_ROOT/oracle_proxy/proxy_server.py"
GAME_CLIENT="$PROJECT_ROOT/actor_n/game.py"
ACTOR="$PROJECT_ROOT/actor_n/actor.py"
LEARNER="$PROJECT_ROOT/learner_n/learner.py"

# Log directory
LOG_DIR="$PROJECT_ROOT/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "  DanZero Oracle Training Launcher"
echo "========================================"
echo "Log directory: $LOG_DIR"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up processes..."
    pkill -f "proxy_server.py" 2>/dev/null || true
    pkill -f "actor_n/game.py" 2>/dev/null || true
    pkill -f "actor_n/actor.py" 2>/dev/null || true
    pkill -f "learner_n/learner.py" 2>/dev/null || true
    pkill -f "guandan_server" 2>/dev/null || true
    echo "Cleanup complete."
}

trap cleanup EXIT

# Check if required files exist
echo "[1/5] Checking required files..."
for f in "$GUANDAN_SERVER" "$ORACLE_PROXY" "$GAME_CLIENT" "$ACTOR" "$LEARNER"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done
echo "      All required files found."

# Start Guandan game server
echo ""
echo "[2/6] Starting Guandan game server on port $GAME_SERVER_PORT..."
cd "$PROJECT_ROOT"
# 参数是运行的局数，设置一个很大的数
./guandan_server 1000000000000 > "$LOG_DIR/game_server.log" 2>&1 &
GAME_SERVER_PID=$!
echo "      PID: $GAME_SERVER_PID"
sleep 2

# Start Oracle proxy server
echo ""
echo "[3/5] Starting Oracle proxy server on port $ORACLE_PROXY_PORT..."
cd "$PROJECT_ROOT/oracle_proxy"
python proxy_server.py > "$LOG_DIR/oracle_proxy.log" 2>&1 &
PROXY_PID=$!
echo "      PID: $PROXY_PID"
sleep 1

# Start Oracle learner
echo ""
echo "[4/5] Starting Oracle learner..."
cd "$PROJECT_ROOT/learner_n"
python learner.py \
    --data_port $LEARNER_DATA_PORT \
    --param_port $LEARNER_PARAM_PORT \
    --exp_path "$LOG_DIR/learner" \
    > "$LOG_DIR/learner.log" 2>&1 &
LEARNER_PID=$!
echo "      PID: $LEARNER_PID"
sleep 2

# Start Oracle actors
echo ""
echo "[5/5] Starting Oracle actors..."
cd "$PROJECT_ROOT/actor_n"
python actor.py \
    --ip 127.0.0.1 \
    --data_port $LEARNER_DATA_PORT \
    --param_port $LEARNER_PARAM_PORT \
    --exp_path "$LOG_DIR/actor" \
    > "$LOG_DIR/actor.log" 2>&1 &
ACTOR_PID=$!
echo "      PID: $ACTOR_PID"
sleep 1

# Start Oracle game clients (4 players - game_oracle.py spawns 4 processes internally)
echo ""
echo "[6/6] Starting Oracle game clients..."
cd "$PROJECT_ROOT/actor_n"
# game.py 内部会启动4个进程，不需要循环启动
python game.py \
    --server_port $ORACLE_PROXY_PORT \
    --oracle \
    > "$LOG_DIR/game_client.log" 2>&1 &
GAME_CLIENT_PID=$!
echo "      Game clients started (PID: $GAME_CLIENT_PID)"

echo ""
echo "========================================"
echo "  All components started!"
echo "========================================"
echo ""
echo "Log files:"
echo "  - Learner: $LOG_DIR/learner.log"
echo "  - Actor:   $LOG_DIR/actor.log"
echo "  - Proxy:   $LOG_DIR/oracle_proxy.log"
echo "  - Game:    $LOG_DIR/game_client.log"
echo ""
echo "Monitoring logs (Ctrl+C to stop)..."
echo ""

# Monitor both learner and actor logs with labels
tail -f "$LOG_DIR/learner.log" "$LOG_DIR/actor.log" 2>/dev/null | while read line; do
    echo "$line"
done
