#!/bin/bash
# ==========================================
#  Stop All Oracle Training Processes
# ==========================================

echo "Stopping all DanZero Oracle training processes..."

# Kill processes by name
pkill -f "guandan_server" 2>/dev/null
pkill -f "proxy_server.py" 2>/dev/null  
pkill -f "actor.py" 2>/dev/null
pkill -f "learner.py" 2>/dev/null
pkill -f "game.py" 2>/dev/null

sleep 2

# Verify all stopped
remaining=$(ps aux | grep -E "guandan_server|proxy_server|actor\.py|learner\.py|game\.py" | grep -v grep | wc -l)

if [ "$remaining" -eq 0 ]; then
    echo "✅ All training processes stopped successfully!"
else
    echo "⚠️  Some processes may still be running. Force killing..."
    pkill -9 -f "guandan_server" 2>/dev/null
    pkill -9 -f "proxy_server.py" 2>/dev/null
    pkill -9 -f "actor.py" 2>/dev/null
    pkill -9 -f "learner.py" 2>/dev/null
    pkill -9 -f "game.py" 2>/dev/null
    sleep 1
    echo "✅ Force kill completed!"
fi

echo ""
echo "Remaining related processes:"
ps aux | grep -E "guandan|proxy|actor|learner" | grep -v grep | grep -v "tail" || echo "  (none)"
