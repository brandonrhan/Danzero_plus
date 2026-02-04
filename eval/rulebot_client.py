#!/usr/bin/env python3
"""
RuleBot 包装器 - 连接到指定端口

用于评估时连接到 Oracle Proxy (23457) 而不是直接连接 guandan_server (23456)
"""

import os
import sys
from argparse import ArgumentParser

# 添加 ai4 路径
AI4_DIR = "/mnt/c/Users/admin/Documents/GitHub/Danzero_plus/wintest/ai4"
sys.path.insert(0, AI4_DIR)

from state import State
from action import Action
import json
from ws4py.client.threadedclient import WebSocketClient


parser = ArgumentParser()
parser.add_argument('--position', type=int, required=True,
                    help='Player position (1 or 3)')
parser.add_argument('--server_port', type=int, default=23457,
                    help='Server port to connect to')


class RuleBotClient(WebSocketClient):
    """RuleBot 客户端"""

    def __init__(self, url, position):
        super().__init__(url)
        self.position = position
        self.state = State(f"client{position}")
        self.action = Action(f"client{position}")
        self.game_count = 0
        self.wins = 0

    def opened(self):
        print(f"[RuleBot Pos {self.position}] Connected")

    def closed(self, code, reason=None):
        print(f"[RuleBot Pos {self.position}] Closed: {code} {reason}")
        if self.game_count > 0:
            print(f"[RuleBot Pos {self.position}] Games: {self.game_count}, Wins: {self.wins}")

    def received_message(self, message):
        message = json.loads(str(message))
        self.state.parse(message)
        
        # 处理游戏结束
        if message.get('type') == 'notify' and message.get('stage') == 'episodeOver':
            self.game_count += 1
            reward = message.get('reward', 0)
            if reward > 0:
                self.wins += 1
            print(f"[RuleBot Pos {self.position}] Game {self.game_count}: Reward={reward}")
        
        if "actionList" in message:
            act_index = self.action.rule_parse(
                message,
                self.state._myPos,
                self.state.remain_cards,
                self.state.history,
                self.state.remain_cards_classbynum,
                self.state.pass_num,
                self.state.my_pass_num,
                self.state.tribute_result
            )
            self.send(json.dumps({"actIndex": act_index}))


def main():
    args, _ = parser.parse_known_args()
    
    url = f'ws://127.0.0.1:{args.server_port}/game/client{args.position}'
    print(f"[RuleBot Pos {args.position}] Connecting to {url}")
    
    client = RuleBotClient(url, args.position)
    client.connect()
    client.run_forever()


if __name__ == '__main__':
    main()
