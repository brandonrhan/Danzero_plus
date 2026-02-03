#!/bin/bash
# Oracle Training Monitor
# 实时监控训练状态

# 找最新的日志目录（只找目录，不找文件）
LOG_BASE="/mnt/c/Users/admin/Documents/GitHub/danzero_oracle/logs"
LATEST_LOG=$(ls -dt "$LOG_BASE"/*/ 2>/dev/null | head -1 | xargs -r basename)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 没有找到日志目录！请先启动训练。"
    exit 1
fi

LOG_DIR="$LOG_BASE/$LATEST_LOG"

echo "========================================"
echo "  Oracle Training Monitor"
echo "  Log Dir: $LATEST_LOG"
echo "========================================"
echo ""

while true; do
    clear
    echo "========================================"
    echo "  Oracle Training Monitor - $(date '+%H:%M:%S')"
    echo "  Log Dir: $LATEST_LOG"
    echo "========================================"
    
    # Learner 状态
    echo ""
    echo "📊 LEARNER 状态"
    echo "----------------------------------------"
    if [ -f "$LOG_DIR/learner/log/progress.csv" ]; then
        TRAIN_COUNT=$(($(wc -l < "$LOG_DIR/learner/log/progress.csv") - 1))
        echo "训练次数: $TRAIN_COUNT"
        echo ""
        echo "最近5次训练 (loss, values):"
        tail -5 "$LOG_DIR/learner/log/progress.csv" | while IFS=',' read -r loss values; do
            if [ "$loss" != "loss" ]; then
                printf "  Loss: %8.4f | Q值: %8.4f\n" "$loss" "$values"
            fi
        done
    else
        echo "等待训练数据... (检查 $LOG_DIR/learner/log/progress.csv)"
    fi
    
    # Actor 状态
    echo ""
    echo "🎮 ACTOR 状态 (每个Client的对局数和步数)"
    echo "----------------------------------------"
    TOTAL_GAMES=0
    for i in 0 1 2 3; do
        if [ -f "$LOG_DIR/actor/Client$i/log/progress.csv" ]; then
            EP_COUNT=$(($(wc -l < "$LOG_DIR/actor/Client$i/log/progress.csv") - 1))
            if [ $EP_COUNT -gt 0 ]; then
                TOTAL_GAMES=$((TOTAL_GAMES + EP_COUNT))
                LAST_STEPS=$(tail -1 "$LOG_DIR/actor/Client$i/log/progress.csv")
                echo "  Actor $i: $EP_COUNT 局完成, 最近步数: $LAST_STEPS"
            else
                echo "  Actor $i: 等待对局..."
            fi
        else
            echo "  Actor $i: 日志文件未创建"
        fi
    done
    echo "  ─────────────────────────────────────"
    echo "  总对局数: $TOTAL_GAMES"
    
    # 最新日志输出
    echo ""
    echo "📝 LEARNER.LOG 最新输出"
    echo "----------------------------------------"
    if [ -f "$LOG_DIR/learner.log" ]; then
        tail -8 "$LOG_DIR/learner.log" 2>/dev/null | head -8
    else
        echo "  等待日志..."
    fi
    
    echo ""
    echo "📝 ACTOR.LOG 最新输出"
    echo "----------------------------------------"
    if [ -f "$LOG_DIR/actor.log" ]; then
        tail -5 "$LOG_DIR/actor.log" 2>/dev/null
    else
        echo "  等待日志..."
    fi
    
    # 训练速度统计
    echo ""
    echo "⏱️ 训练速度"
    echo "----------------------------------------"
    if [ -f "$LOG_DIR/learner/log/progress.csv" ]; then
        START_TIME=$(stat -c %Y "$LOG_DIR" 2>/dev/null || echo 0)
        if [ "$START_TIME" != "0" ]; then
            NOW=$(date +%s)
            ELAPSED=$((NOW - START_TIME))
            TRAIN_COUNT=$(($(wc -l < "$LOG_DIR/learner/log/progress.csv") - 1))
            if [ $ELAPSED -gt 0 ] && [ $TRAIN_COUNT -gt 0 ]; then
                RATE=$(echo "scale=2; $TRAIN_COUNT * 60 / $ELAPSED" | bc 2>/dev/null || echo "N/A")
                GAME_RATE=$(echo "scale=2; $TOTAL_GAMES * 60 / $ELAPSED" | bc 2>/dev/null || echo "N/A")
                ELAPSED_MIN=$((ELAPSED / 60))
                ELAPSED_SEC=$((ELAPSED % 60))
                echo "  运行时间: ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
                echo "  训练速度: ~$RATE 次/分钟"
                echo "  对局速度: ~$GAME_RATE 局/分钟"
            fi
        fi
    fi
    
    # 进程状态
    echo ""
    echo "🔧 进程状态"
    echo "----------------------------------------"
    LEARNER_CNT=$(pgrep -f "learner.py" | wc -l)
    ACTOR_CNT=$(pgrep -f "actor.py" | wc -l)
    GAME_CNT=$(pgrep -f "game.py" | wc -l)
    PROXY_CNT=$(pgrep -f "proxy_server.py" | wc -l)
    SERVER_CNT=$(pgrep -f "guandan_server" | wc -l)
    echo "  Learner: $LEARNER_CNT | Actor: $ACTOR_CNT | Game: $GAME_CNT | Proxy: $PROXY_CNT | Server: $SERVER_CNT"
    
    echo ""
    echo "========================================" 
    echo "按 Ctrl+C 退出监控 | 每3秒刷新"
    sleep 3
done
