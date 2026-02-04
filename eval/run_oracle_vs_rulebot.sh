#!/bin/bash
#
# DanZero Oracle vs RuleBot 评估脚本
#
# 架构:
#   guandan_server (23456) <- Oracle Proxy (23457) <- Oracle Clients (位置0,2)
#                                                  <- RuleBot (位置1,3)
#
# Oracle 特性:
#   - 使用 729 维状态输入 (567 原始 + 162 oracle信息)
#   - 从 oracle_proxy 接收所有玩家完整手牌 (allHandCards)
#   - 加载训练产生的 oracle ckpt 模型
#
# 用法: ./run_oracle_vs_rulebot.sh [num_games] [model_path]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORACLE_DIR="$(dirname "$SCRIPT_DIR")"
PROXY_DIR="$ORACLE_DIR/oracle_proxy"
AI4_DIR="$ORACLE_DIR/../Danzero_plus/wintest/ai4"
GUANDAN_SERVER="$ORACLE_DIR/guandan_server"

NUM_GAMES=${1:-10}
MODEL_PATH=${2:-"$ORACLE_DIR/logs/20260202_022157/actor/Client0/ckpt"}

# 使用 tf115 环境
PYTHON="/home/brandonrhan/miniconda/envs/tf115/bin/python"

# 端口配置
SERVER_PORT=23456
PROXY_PORT=23457
ACTOR_BASE_PORT=6000

LOG_DIR="$ORACLE_DIR/eval_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "DanZero Oracle vs RuleBot 评估"
echo "=============================================="
echo "游戏局数: $NUM_GAMES"
echo "模型路径: $MODEL_PATH"
echo "Python: $PYTHON"
echo ""
echo "Oracle 特性:"
echo "  - 729 维状态输入 (含 allHandCards)"
echo "  - 使用 oracle_proxy 转发完整手牌信息"
echo ""
echo "位置分配:"
echo "  - 位置 0, 2: DanZero Oracle (队友)"
echo "  - 位置 1, 3: RuleBot ai4 (队友)"
echo ""

# 检查端口
check_port() {
    if ss -tlnp 2>/dev/null | grep -q ":$1 "; then
        echo "ERROR: 端口 $1 被占用!"
        return 1
    fi
    return 0
}

echo "检查端口..."
check_port $SERVER_PORT || exit 1
check_port $PROXY_PORT || exit 1
echo "端口可用"
echo ""

# 清理函数
cleanup() {
    echo ""
    echo "清理进程..."
    pkill -f "$GUANDAN_SERVER" 2>/dev/null || true
    pkill -f "proxy_server.py" 2>/dev/null || true
    pkill -f "oracle_eval_actor.py" 2>/dev/null || true
    pkill -f "oracle_eval_client.py" 2>/dev/null || true
    pkill -f "rulebot_client.py" 2>/dev/null || true
    sleep 1
    echo "清理完成"
}

trap cleanup EXIT
cleanup

# 1. 启动游戏服务器
echo "1. 启动 guandan_server ($NUM_GAMES 局)..."
cd "$ORACLE_DIR"
"$GUANDAN_SERVER" $NUM_GAMES > "$LOG_DIR/server_$TIMESTAMP.log" 2>&1 &
SERVER_PID=$!
sleep 2

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: guandan_server 启动失败!"
    cat "$LOG_DIR/server_$TIMESTAMP.log"
    exit 1
fi
echo "   Server 已启动 (PID: $SERVER_PID, 端口: $SERVER_PORT)"

# 2. 启动 Oracle Proxy (转发 allHandCards)
echo "2. 启动 Oracle Proxy..."
cd "$PROXY_DIR"
$PYTHON proxy_server.py \
    --proxy-port $PROXY_PORT \
    --upstream-port $SERVER_PORT \
    > "$LOG_DIR/proxy_$TIMESTAMP.log" 2>&1 &
PROXY_PID=$!
sleep 2

if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo "ERROR: Oracle Proxy 启动失败!"
    cat "$LOG_DIR/proxy_$TIMESTAMP.log"
    exit 1
fi
echo "   Proxy 已启动 (PID: $PROXY_PID, 端口: $PROXY_PORT)"

# 3. 启动 Oracle Actor (加载 729 维 oracle 模型)
echo "3. 启动 Oracle Actor (729维模型)..."
cd "$SCRIPT_DIR"
$PYTHON oracle_eval_actor.py \
    --model_path "$MODEL_PATH" \
    --positions "0,2" \
    --base_port $ACTOR_BASE_PORT \
    > "$LOG_DIR/actor_$TIMESTAMP.log" 2>&1 &
ACTOR_PID=$!
sleep 10  # 增加等待时间让 TensorFlow 完全初始化

if ! kill -0 $ACTOR_PID 2>/dev/null; then
    echo "ERROR: Oracle Actor 启动失败!"
    cat "$LOG_DIR/actor_$TIMESTAMP.log"
    exit 1
fi
echo "   Actor 已启动 (PID: $ACTOR_PID)"
echo "   等待 Actor 完全初始化..."
sleep 2

# 4. 按顺序启动客户端 (连接到 Proxy)
echo "4. 启动客户端..."

# 位置0: Oracle Client (使用 729 维状态)
echo "   启动 Oracle Client (位置 0, 729维)..."
cd "$SCRIPT_DIR"
$PYTHON oracle_eval_client.py \
    --position 0 \
    --server_port $PROXY_PORT \
    --action_port $ACTOR_BASE_PORT \
    > "$LOG_DIR/client0_$TIMESTAMP.log" 2>&1 &
sleep 1

# 位置1: RuleBot (连接到 proxy)
echo "   启动 RuleBot (位置 1)..."
$PYTHON rulebot_client.py \
    --position 1 \
    --server_port $PROXY_PORT \
    > "$LOG_DIR/rulebot1_$TIMESTAMP.log" 2>&1 &
sleep 1

# 位置2: Oracle Client
echo "   启动 Oracle Client (位置 2, 729维)..."
$PYTHON oracle_eval_client.py \
    --position 2 \
    --server_port $PROXY_PORT \
    --action_port $ACTOR_BASE_PORT \
    > "$LOG_DIR/client2_$TIMESTAMP.log" 2>&1 &
sleep 1

# 位置3: RuleBot (连接到 proxy)
echo "   启动 RuleBot (位置 3)..."
$PYTHON rulebot_client.py \
    --position 3 \
    --server_port $PROXY_PORT \
    > "$LOG_DIR/rulebot3_$TIMESTAMP.log" 2>&1 &
sleep 1

echo ""
echo "所有进程已启动!"
echo "日志目录: $LOG_DIR"
echo ""
echo "等待游戏完成..."

# 等待服务器完成
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "评估完成!"
echo "=============================================="

# 查看日志
echo ""
echo "=== 服务器日志 ==="
tail -20 "$LOG_DIR/server_$TIMESTAMP.log" 2>/dev/null || true

echo ""
echo "=== Client 0 日志 ==="
tail -10 "$LOG_DIR/client0_$TIMESTAMP.log" 2>/dev/null || true

echo ""
echo "=== Client 2 日志 ==="
tail -10 "$LOG_DIR/client2_$TIMESTAMP.log" 2>/dev/null || true

echo ""
echo "Done!"
