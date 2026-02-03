#!/bin/bash
#
# DanZero Oracle vs RuleBot (ai4) 测试脚本
# Oracle 使用 729 维状态输入（包含所有玩家手牌信息）
#
# 用法: ./test_oracle_vs_rulebot.sh [num_games]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DANZERO_DIR="$SCRIPT_DIR/danzero"
AI4_DIR="$SCRIPT_DIR/ai4"
ORACLE_ROOT="$(dirname "$SCRIPT_DIR")"
DANSERVER="$SCRIPT_DIR/torch/danserver"
NUM_GAMES=${1:-10}

# 使用 tf115 环境 (有 tensorflow 和正确版本的 pyarrow)
PYTHON="/home/brandonrhan/miniconda/envs/tf115/bin/python"

echo "=============================================="
echo "DanZero Oracle vs RuleBot (ai4)"
echo "=============================================="
echo "Games: $NUM_GAMES"
echo "Python: $PYTHON"
echo ""

# 检查端口
if ss -tlnp 2>/dev/null | grep -q ":23456 "; then
    echo "ERROR: Port 23456 is in use!"
    exit 1
fi

# 清理函数
cleanup() {
    echo "Cleaning up..."
    pkill -f "$DANSERVER" 2>/dev/null || true
    pkill -f "danzero/client" 2>/dev/null || true
    pkill -f "danzero/actor" 2>/dev/null || true
    pkill -f "ai4/client" 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT
cleanup

# 检查 danserver 是否存在
if [ ! -f "$DANSERVER" ]; then
    echo "ERROR: danserver not found at $DANSERVER"
    echo "Trying alternative locations..."
    if [ -f "$ORACLE_ROOT/guandan_server" ]; then
        DANSERVER="$ORACLE_ROOT/guandan_server"
        echo "Using: $DANSERVER"
    else
        exit 1
    fi
fi

# 启动 server
echo "Starting server for $NUM_GAMES games..."
"$DANSERVER" $NUM_GAMES &
SERVER_PID=$!
sleep 1

# 启动 actor (为位置0和2提供推理)
echo "Starting Oracle actor..."
cd "$DANZERO_DIR"
$PYTHON actor.py &
ACTOR_PID=$!
sleep 2

# 按顺序启动客户端
echo "Starting clients..."

# 位置 0: Oracle
cd "$DANZERO_DIR"
$PYTHON client0.py &
sleep 0.3

# 位置 1: RuleBot
cd "$AI4_DIR"
$PYTHON client1.py &
sleep 0.3

# 位置 2: Oracle
cd "$DANZERO_DIR"
$PYTHON client2.py &
sleep 0.3

# 位置 3: RuleBot
cd "$AI4_DIR"
$PYTHON client3.py &
sleep 0.3

echo ""
echo "All processes started. Waiting for games..."
echo ""

# 等待服务器完成
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "Testing complete!"
echo "=============================================="
