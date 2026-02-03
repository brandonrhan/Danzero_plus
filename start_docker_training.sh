#!/bin/bash
# 启动并行 Docker 训练
# 用法: ./start_docker_training.sh [num_instances]

set -e

NUM_INSTANCES=${1:-4}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  Docker Parallel Training Launcher"
echo "========================================"
echo "Number of instances: $NUM_INSTANCES"
echo ""

# 构建镜像
echo "[1/3] Building Docker images..."
docker compose -f docker-compose.parallel.yaml build

# 根据实例数量选择要启动的服务
SERVICES="learner"
for ((i=0; i<NUM_INSTANCES; i++)); do
    SERVICES="$SERVICES instance_$i"
done

echo ""
echo "[2/3] Starting services: $SERVICES"
docker compose -f docker-compose.parallel.yaml up -d $SERVICES

echo ""
echo "[3/3] Showing logs..."
echo "Press Ctrl+C to stop viewing logs (containers will keep running)"
echo ""

docker compose -f docker-compose.parallel.yaml logs -f learner
