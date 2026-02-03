#!/bin/bash
# Docker instance training script
# 每个容器运行一个完整的训练实例：server + proxy + actors + game clients
# 连接到外部共享的 learner

set -e

# 从环境变量获取配置
LEARNER_IP=${LEARNER_IP:-172.17.0.1}  # Docker host IP
LEARNER_DATA_PORT=${LEARNER_DATA_PORT:-5000}
LEARNER_PARAM_PORT=${LEARNER_PARAM_PORT:-5001}
INSTANCE_ID=${INSTANCE_ID:-0}
NUM_GAMES=${NUM_GAMES:-1000000000}

echo "========================================"
echo "  Oracle Training Instance $INSTANCE_ID"
echo "========================================"
echo "Learner: $LEARNER_IP:$LEARNER_DATA_PORT/$LEARNER_PARAM_PORT"
echo ""

# 启动 guandan_server
echo "[1/4] Starting game server..."
cd /app
./guandan_server $NUM_GAMES &
SERVER_PID=$!
sleep 2

# 启动 oracle proxy
echo "[2/4] Starting oracle proxy..."
cd /app/oracle_proxy
python proxy_server.py &
PROXY_PID=$!
sleep 1

# 启动 actors (连接到外部 learner)
echo "[3/4] Starting actors..."
cd /app/actor_n
python actor.py \
    --ip $LEARNER_IP \
    --data_port $LEARNER_DATA_PORT \
    --param_port $LEARNER_PARAM_PORT \
    --exp_path /app/logs/actor \
    --base_port 6000 \
    &
ACTOR_PID=$!
sleep 2

# 启动 game clients
echo "[4/4] Starting game clients..."
cd /app/actor_n
python game.py \
    --server_port 23457 \
    --action_port 6000 \
    --oracle \
    &
GAME_PID=$!

echo ""
echo "All components started!"
echo "  Server PID: $SERVER_PID"
echo "  Proxy PID: $PROXY_PID"
echo "  Actor PID: $ACTOR_PID"
echo "  Game PID: $GAME_PID"
echo ""

# 等待任意进程退出
wait -n
echo "A process exited, shutting down..."
