#!/bin/bash
#
# DanZero(torch) vs RuleBot(ai4) 测试脚本
# 已适配当前环境
#
# 用法: ./testmodel_local.sh [num_games]
#
# 注意: 需要端口 23456 空闲！如果 danzero_oracle 在训练，需要先停止。
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WINTEST_DIR="$(dirname "$SCRIPT_DIR")"
DANSERVER="$SCRIPT_DIR/danserver"
AI4_DIR="$WINTEST_DIR/ai4"
NUM_GAMES=${1:-100}

echo "=============================================="
echo "DanZero vs RuleBot(ai4) Test"
echo "=============================================="
echo "Games: $NUM_GAMES"
echo ""

# 检查端口
if ss -tlnp 2>/dev/null | grep -q ":23456 "; then
    echo "ERROR: Port 23456 is in use!"
    echo "If danzero_oracle is training, run: cd /mnt/c/Users/admin/Documents/GitHub/danzero_oracle && bash stop_training.sh"
    exit 1
fi

# 清理
cleanup() {
    pkill -f "$DANSERVER" 2>/dev/null || true
    pkill -f "$AI4_DIR/client" 2>/dev/null || true
    pkill -f "$SCRIPT_DIR/client" 2>/dev/null || true
    pkill -f "$SCRIPT_DIR/actor" 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT
cleanup

# 启动 server
echo "Starting danserver..."
nohup "$DANSERVER" $NUM_GAMES >/dev/null 2>&1 &
sleep 0.5

# 启动客户端
# 位置: client1 -> pos 1, client3 -> pos 3 (DanZero队伍)
#       ai4/client1 -> pos 0, ai4/client3 -> pos 2 (RuleBot队伍)

echo "Starting RuleBot client1 (pos 0)..."
nohup python "$AI4_DIR/client1.py" >/dev/null 2>&1 &
sleep 0.5

echo "Starting DanZero client1 (pos 1)..."
nohup python "$SCRIPT_DIR/client1.py" --resfile "$SCRIPT_DIR/results_danzero_vs_rulebot.log" >/dev/null 2>&1 &
sleep 0.5

echo "Starting RuleBot client3 (pos 2)..."
nohup python "$AI4_DIR/client3.py" >/dev/null 2>&1 &
sleep 0.5

echo "Starting DanZero client3 (pos 3)..."
nohup python "$SCRIPT_DIR/client3.py" >/dev/null 2>&1 &
sleep 0.5

echo "Starting DanZero actor..."
nohup python "$SCRIPT_DIR/actor.py" --iter 0 >/dev/null 2>&1 &
sleep 1

echo ""
echo "All processes started. Waiting for $NUM_GAMES games..."
echo "Results will be saved to: $SCRIPT_DIR/results_danzero_vs_rulebot.log"
echo ""

# 等待完成
wait

echo ""
echo "=============================================="
echo "Games completed! Results:"
echo "=============================================="
if [ -f "$SCRIPT_DIR/results_danzero_vs_rulebot.log" ]; then
    cat "$SCRIPT_DIR/results_danzero_vs_rulebot.log"
    
    # 统计
    TOTAL=$(wc -l < "$SCRIPT_DIR/results_danzero_vs_rulebot.log" 2>/dev/null || echo 0)
    WINS=$(grep -cE "Reward: [123]" "$SCRIPT_DIR/results_danzero_vs_rulebot.log" 2>/dev/null || echo 0)
    
    if [ "$TOTAL" -gt 0 ]; then
        WIN_RATE=$(echo "scale=1; $WINS * 100 / $TOTAL" | bc)
        echo ""
        echo "DanZero Win Rate: $WINS / $TOTAL = ${WIN_RATE}%"
    fi
fi

cleanup
echo "Done!"
