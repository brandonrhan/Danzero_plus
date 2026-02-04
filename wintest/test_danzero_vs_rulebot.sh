#!/bin/bash
#
# DanZero (Q-Network) vs RuleBot (ai4) 测试脚本
# 使用 danzero 目录的纯 Q-Network 版本（不需要 PPO 模型）
#
# 用法: ./test_danzero_vs_rulebot.sh [num_games]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DANZERO_DIR="$SCRIPT_DIR/danzero"
AI4_DIR="$SCRIPT_DIR/ai4"
DANSERVER="$SCRIPT_DIR/torch/danserver"
NUM_GAMES=${1:-10}

# 使用 tf115 环境 (有 tensorflow 和正确版本的 pyarrow)
PYTHON="/home/brandonrhan/miniconda/envs/tf115/bin/python"

echo "=============================================="
echo "DanZero (Q-Network) vs RuleBot (ai4)"
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

# 启动 server
echo "Starting danserver for $NUM_GAMES games..."
"$DANSERVER" $NUM_GAMES &
SERVER_PID=$!
sleep 1

# 检查 server 是否启动
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: danserver failed to start!"
    exit 1
fi
echo "Server started (PID: $SERVER_PID)"

# 按顺序启动客户端 (连接顺序决定位置)
# 位置 0: DanZero client0
# 位置 1: RuleBot client1  
# 位置 2: DanZero client2
# 位置 3: RuleBot client3

echo "Starting DanZero client0 (position 0)..."
cd "$DANZERO_DIR"
$PYTHON client0.py &
sleep 0.3

echo "Starting RuleBot client1 (position 1)..."
cd "$AI4_DIR"
$PYTHON client1.py &
sleep 0.3

echo "Starting DanZero client2 (position 2)..."
cd "$DANZERO_DIR"
$PYTHON client2.py &
sleep 0.3

echo "Starting RuleBot client3 (position 3)..."
cd "$AI4_DIR"
$PYTHON client3.py &
sleep 0.3

# 启动 DanZero actor (处理位置 0 和 2 的决策)
echo "Starting DanZero actor..."
cd "$DANZERO_DIR"
$PYTHON actor.py &
sleep 1

echo ""
echo "All processes started!"
echo "Waiting for $NUM_GAMES games to complete..."
echo ""

# 等待 server 完成
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "Test completed!"
echo "=============================================="

cleanup
