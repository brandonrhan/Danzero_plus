#!/usr/bin/env python3
"""
DanZero 客户端 wrapper - 连接到 client1 endpoint (获得 myPos=3，因为是第4个连接)
基于 client3.py（使用相同的 actor），只修改连接的 endpoint
"""
import os
import sys

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 修改 client3.py 中的 endpoint
import client3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--action_port', type=int, default=6001)
args = parser.parse_args()

# 使用 client1 endpoint（第4个连接 → myPos=3）
ws = client3.ExampleClient('ws://127.0.0.1:23456/game/client1', args)
ws.connect()
ws.run_forever()
