#!/usr/bin/env python3
"""
DanZero 客户端 - 连接到 client3 endpoint (获得 myPos=1)
这是原始 client3.py 的简单入口点
"""
import os
import sys

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import client3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--action_port', type=int, default=6001)
args = parser.parse_args()

# 使用 client3 endpoint (获得 myPos=1)
ws = client3.ExampleClient('ws://127.0.0.1:23456/game/client3', args)
ws.connect()
ws.run_forever()
