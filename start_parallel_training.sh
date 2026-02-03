#!/bin/bash
# Parallel Oracle Training Script for DanZero
# 
# This script starts multiple game instances in parallel to speed up data generation
# Each game instance uses its own set of ports:
#   Instance 0: Server 23456, Proxy 23457, Actor ports 6000-6003
#   Instance 1: Server 23466, Proxy 23467, Actor ports 6010-6013
#   Instance 2: Server 23476, Proxy 23477, Actor ports 6020-6023
#   ...
#
# All instances share the same learner (ports 5000-5001)

set -e

# Configuration
NUM_PARALLEL=${1:-2}  # Default 2 parallel games, can be passed as argument
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Activate conda environment with TensorFlow 1.15
source ~/miniconda/etc/profile.d/conda.sh
conda activate tf115
echo "Using Python: $(which python)"

# Learner ports (shared)
LEARNER_DATA_PORT=5000
LEARNER_PARAM_PORT=5001

# Paths
GUANDAN_SERVER="$PROJECT_ROOT/guandan_server"  # 原始Python版本
ORACLE_PROXY="$PROJECT_ROOT/oracle_proxy/proxy_server.py"
GAME_CLIENT="$PROJECT_ROOT/actor_n/game.py"
ACTOR="$PROJECT_ROOT/actor_n/actor.py"
LEARNER="$PROJECT_ROOT/learner_n/learner.py"

# Log directory
LOG_DIR="$PROJECT_ROOT/logs/$(date +%Y%m%d_%H%M%S)_parallel_${NUM_PARALLEL}"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "  DanZero Parallel Oracle Training"
echo "========================================"
echo "Parallel instances: $NUM_PARALLEL"
echo "Log directory: $LOG_DIR"
echo ""

# Function to cleanup on exit (disabled for background running)
# cleanup() {
#     echo ""
#     echo "Cleaning up processes..."
#     pkill -f "proxy_server.py" 2>/dev/null || true
#     pkill -f "actor_n/game.py" 2>/dev/null || true
#     pkill -f "actor_n/actor.py" 2>/dev/null || true
#     pkill -f "learner_n/learner.py" 2>/dev/null || true
#     pkill -f "guandan_server" 2>/dev/null || true
#     echo "Cleanup complete."
# }

# trap cleanup EXIT

# Start learner (shared by all instances)
echo "[1/3] Starting shared learner..."
cd "$PROJECT_ROOT/learner_n"
python learner.py \
    --data_port $LEARNER_DATA_PORT \
    --param_port $LEARNER_PARAM_PORT \
    --exp_path "$LOG_DIR/learner" \
    > "$LOG_DIR/learner.log" 2>&1 &
LEARNER_PID=$!
echo "      Learner PID: $LEARNER_PID"
sleep 2

# Start multiple game instances with their own actors
echo ""
echo "[2/3] Starting $NUM_PARALLEL parallel game instances..."

for ((i=0; i<NUM_PARALLEL; i++)); do
    GAME_SERVER_PORT=$((23456 + i*10))
    ORACLE_PROXY_PORT=$((23457 + i*10))
    ACTOR_BASE_PORT=$((6000 + i*10))
    
    echo ""
    echo "  [Instance $i] Ports: Server=$GAME_SERVER_PORT, Proxy=$ORACLE_PROXY_PORT, Actor=$ACTOR_BASE_PORT-$((ACTOR_BASE_PORT+3))"
    
    # Start game server with port parameter
    cd "$PROJECT_ROOT"
    $GUANDAN_SERVER -i 127.0.0.1 -p $GAME_SERVER_PORT > "$LOG_DIR/game_server_$i.log" 2>&1 &
    echo "    Game server started (PID: $!)"
    sleep 1
    
    # Start proxy server with custom ports
    cd "$PROJECT_ROOT/oracle_proxy"
    python proxy_server.py --proxy-port $ORACLE_PROXY_PORT --upstream-port $GAME_SERVER_PORT > "$LOG_DIR/proxy_$i.log" 2>&1 &
    echo "    Proxy server started (PID: $!)"
    sleep 0.5
    
    # Start actors for this instance (each instance needs 4 actors)
    cd "$PROJECT_ROOT/actor_n"
    python actor.py \
        --ip 127.0.0.1 \
        --data_port $LEARNER_DATA_PORT \
        --param_port $LEARNER_PARAM_PORT \
        --exp_path "$LOG_DIR/actor_$i" \
        --base_port $ACTOR_BASE_PORT \
        > "$LOG_DIR/actor_$i.log" 2>&1 &
    echo "    Actors started (PID: $!, ports $ACTOR_BASE_PORT-$((ACTOR_BASE_PORT+3)))"
    sleep 1
    
    # Start game clients with custom ports
    cd "$PROJECT_ROOT/actor_n"
    python game.py \
        --server_port $ORACLE_PROXY_PORT \
        --action_port $ACTOR_BASE_PORT \
        --oracle \
        > "$LOG_DIR/game_client_$i.log" 2>&1 &
    echo "    Game clients started (PID: $!)"
    sleep 0.5
done

echo ""
echo "========================================"
echo "  All $NUM_PARALLEL instances started!"
echo "========================================"
echo ""
echo "Log files:"
echo "  - Learner: $LOG_DIR/learner.log"
for ((i=0; i<NUM_PARALLEL; i++)); do
    echo "  - Actor $i:   $LOG_DIR/actor_$i.log"
    echo "  - Game $i:  $LOG_DIR/game_client_$i.log"
done
echo ""
echo "Monitoring learner (Ctrl+C to stop)..."
echo ""

# Monitor learner log
tail -f "$LOG_DIR/learner.log"
