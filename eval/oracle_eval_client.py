#!/usr/bin/env python3
"""
DanZero Oracle 评估客户端

基于标准 DanZero client0.py 修改，添加 Oracle (allHandCards) 支持
输入维度: 729 = 513 (标准) + 162 (oracle) + 54 (action)

连接到 Oracle Proxy Server 获取所有玩家手牌
"""

import os
import sys
import json
import time
import warnings
from argparse import ArgumentParser
from functools import reduce
from multiprocessing import Process
from random import randint

import zmq
import numpy as np
from ws4py.client.threadedclient import WebSocketClient
from pyarrow import deserialize, serialize

# 添加 utils 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.utils import *

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = ArgumentParser()
parser.add_argument('--ip', type=str, default='127.0.0.1',
                    help='IP address of actor server')
parser.add_argument('--action_port', type=int, default=6000,
                    help='Actor server base port')
parser.add_argument('--server_port', type=int, default=23457,
                    help='Game server port (23457 for oracle proxy)')
parser.add_argument('--position', type=int, required=True,
                    help='Player position (0 or 2 for oracle)')
parser.add_argument('--resfile', type=str, default='eval_results.log',
                    help='Results file path')


RANK = {
    '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8,
    'T':9, 'J':10, 'Q':11, 'K':12, 'A':13
}

card_value_s2v = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                  'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, 'B': 16, 'R': 17}


def _get_one_hot_array(num_left_cards, max_num_cards, flag):
    if flag == 0:
        one_hot = np.zeros(max_num_cards)
        one_hot[num_left_cards - 1] = 1
    else:
        one_hot = np.zeros(max_num_cards+1)
        one_hot[num_left_cards] = 1
    return one_hot


class OracleEvalClient(WebSocketClient):
    """Oracle 评估客户端 - 基于标准 DanZero client0.py"""
    
    def __init__(self, url, args):
        super().__init__(url)
        self.args = args
        self.mypos = args.position
        self.history_action = {0: [], 1: [], 2: [], 3:[]}
        self.action_seq = []
        self.action_order = []
        self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self.other_left_hands = [2 for _ in range(54)]
        self.flag = 0
        self.over = []
        self.rank = {'self_rank': 1, 'oppo_rank': 1}
        
        # Oracle: 存储所有玩家手牌
        self.all_hand_cards = [[], [], [], []]
        
        # 游戏统计
        self.tongji = {3: 0, 2: 0, 1: 0, -1: 0, -2: 0, -3: 0, 'all': 0}

        # 初始化zmq连接到actor
        self.context = zmq.Context()
        self.context.linger = 0
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30秒超时
        self.socket.setsockopt(zmq.SNDTIMEO, 30000)  # 30秒超时
        actor_port = self.args.action_port + self.args.position
        self.socket.connect(f'tcp://localhost:{actor_port}')
        print(f"[Pos {self.mypos}] Connected to actor at port {actor_port}")
        
    def opened(self):
        print(f"[Pos {self.mypos}] WebSocket connected")

    def closed(self, code, reason=None):
        print(f"[Pos {self.mypos}] Closed: {code} {reason}")
        # 输出最终结果
        if self.tongji['all'] > 0:
            print(f"[Pos {self.mypos}] Final stats: {self.tongji}")
            with open(self.args.resfile, 'a') as f:
                f.write(f"Pos {self.mypos} Final: {self.tongji}\n")

    def received_message(self, message):
        message = json.loads(str(message))
        
        if message['type'] == 'notify':
            # 牌局开始记录位置
            if message['stage'] == 'beginning':
                print(f"[Pos {self.mypos}] curRank: {message['curRank']}")
                self.mypos = message['myPos']
                self.list_cards = message['handCards']
                
            # 记录进贡的牌
            elif message['stage'] == 'tribute':
                self.tribute_result = message['result']
                
            # 在动作序列中记录动作
            elif message['stage'] == 'play':
                self._record_play(message)
                
            # 小局结束
            elif message['stage'] == 'episodeOver':
                self._handle_episode_over(message)
                
        elif message['type'] == 'act':
            self._handle_act(message)

    def _record_play(self, message):
        """记录出牌动作 - 直接复制自 client0.py"""
        just_play = message['curPos']
        action = card2num(message['curAction'][2])
        
        if message['curPos'] != self.mypos:
            for ele in action:
                if 0 <= ele < 54:
                    self.other_left_hands[ele] -= 1
                    
        # 更新剩余牌数
        if action:
            self.remaining[just_play] -= len(action)
                    
        if len(self.over) == 0:
            self.action_order.append(just_play)
            self.action_seq.append(action)
            self.history_action[message['curPos']].append(action)
            
            # 检查是否有人出完
            if self.remaining[just_play] <= 0:
                self.over.append(just_play)
                self.flag = 1
        else:
            # 简化处理：直接记录
            self.action_order.append(just_play)
            self.action_seq.append(action)
            self.history_action[message['curPos']].append(action)
            
            if self.remaining[just_play] <= 0 and just_play not in self.over:
                self.over.append(just_play)
        
        # 更新自己手牌
        if just_play == self.mypos:
            played_cards = message['curAction'][2]
            for card in played_cards:
                if card in self.list_cards:
                    self.list_cards.remove(card)

    def _handle_episode_over(self, message):
        """处理小局结束"""
        self.history_action = {0: [], 1: [], 2: [], 3:[]}
        self.action_seq = []
        self.other_left_hands = [2 for _ in range(54)]
        self.flag = 0
        self.action_order = []
        self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self.over = []
        self.all_hand_cards = [[], [], [], []]
        
        reward = self._get_reward(message)
        self.tongji[reward] += 1
        self.tongji['all'] += 1
        
        print(f"[Pos {self.mypos}] Game {self.tongji['all']}: reward={reward}")

    def _get_reward(self, message):
        team = [self.mypos, (self.mypos + 2) % 4]
        order = message['order']
        rewards = {"1100": 3, "1010": 2, "1001": 1, "0110": -1, "0101": -2, "0011": -3}
        res = ""
        for i in order:
            if i in team:
                res += '1'
            else:
                res += '0'
        return rewards.get(res, 0)

    def proc_universal(self, handCards, cur_rank):
        """处理万能牌标志位 - 复制自 client0.py"""
        res = np.zeros(12, dtype=np.int8)

        if handCards[(cur_rank-1)*4] == 0:
            return res

        res[0] = 1
        rock_flag = 0
        for i in range(4):
            left, right = 0, 5
            temp = [handCards[i + j*4] if i+j*4 != (cur_rank-1)*4 else 0 for j in range(5)]
            while right <= 12:
                zero_num = temp.count(0)
                if zero_num <= 1:
                    rock_flag = 1
                    break
                else:
                    temp.append(handCards[i + right*4] if i+right*4 != (cur_rank-1)*4 else 0)
                    temp.pop(0)
                    left += 1
                    right += 1
            if rock_flag == 1:
                break
        res[1] = rock_flag

        num_count = [0] * 13
        for i in range(4):
            for j in range(13):
                if handCards[i + j*4] != 0 and i + j*4 != (cur_rank-1)*4:
                    num_count[j] += 1
        num_max = max(num_count)
        if num_max >= 6:
            res[2:8] = 1
        elif num_max == 5:
            res[3:8] = 1
        elif num_max == 4:
            res[4:8] = 1
        elif num_max == 3:
            res[5:8] = 1
        elif num_max == 2:
            res[6:8] = 1
        else:
            res[7] = 1
        temp = 0
        for i in range(13):
            if num_count[i] != 0:
                temp += 1
                if i >= 1:
                    if num_count[i] == 2 and num_count[i-1] >= 3 or num_count[i] >= 3 and num_count[i-1] == 2:
                        res[9] = 1
                    elif num_count[i] == 2 and num_count[i-1] == 2:
                        res[11] = 1
                if i >= 2:
                    if num_count[i-2] == 1 and num_count[i-1] >= 2 and num_count[i] >= 2 or \
                        num_count[i-2] >= 2 and num_count[i-1] == 1 and num_count[i] >= 2 or \
                        num_count[i-2] >= 2 and num_count[i-1] >= 2 and num_count[i] == 1:
                        res[10] = 1
            else:
                temp = 0
        if temp >= 4:
            res[8] = 1
        return res

    def _handle_act(self, message):
        """处理 act 消息 - 构建 Oracle 状态"""
        # Oracle: 读取所有玩家手牌
        if 'allHandCards' in message:
            self.all_hand_cards = message['allHandCards']
            # 调试: 确认收到了 oracle 信息 (具体手牌，不仅仅是数量)
            if not hasattr(self, '_oracle_debug_count'):
                self._oracle_debug_count = 0
            if self._oracle_debug_count < 1:
                import sys
                # 打印每个玩家的具体手牌（只显示前5张）
                for i, cards in enumerate(self.all_hand_cards):
                    sample = cards[:5] if len(cards) > 5 else cards
                    print(f"[Pos {self.mypos}] Player {i} cards: {sample}... ({len(cards)} total)", flush=True)
                    sys.stdout.flush()
                self._oracle_debug_count += 1
        else:
            # 没有收到 allHandCards - 这是个问题!
            if not hasattr(self, '_no_oracle_warned'):
                print(f"[Pos {self.mypos}] WARNING: No allHandCards in message! Not using Oracle!")
                self._no_oracle_warned = True
            
        # 只有一个选择时直接发送
        if len(message['actionList']) == 1:
            self.send(json.dumps({"actIndex": 0}))
            return
            
        # 构建 oracle 状态并获取动作
        state = self.prepare(message)
        
        # 传输给决策模块
        self.socket.send(serialize(state).to_buffer())
        # 收到决策
        act_index = deserialize(self.socket.recv())
        # 发送动作
        self.send(json.dumps({"actIndex": int(act_index)}))

    def prepare(self, message):
        """
        构建 729 维 Oracle 状态
        
        标准部分 (513维, 不含动作):
        - my_handcards: 54
        - universal_card_flag: 12
        - other_handcards: 54
        - last_action: 54
        - last_teammate_action: 54
        - down_played_cards: 54
        - teammate_played_cards: 54
        - up_played_cards: 54
        - down_num_cards_left: 28
        - teammate_num_cards_left: 28
        - up_num_cards_left: 28
        - self_rank: 13
        - oppo_rank: 13
        - cur_rank: 13
        
        Oracle 部分 (162维):
        - down_handcards: 54
        - teammate_handcards: 54
        - up_handcards: 54
        
        动作部分 (54维):
        - my_action: 54
        
        总计: 513 + 162 + 54 = 729
        """
        num_legal_actions = message['indexRange'] + 1
        legal_actions = [card2num(i[2]) for i in message['actionList']]
        
        # 自己的手牌, 54维
        my_handcards = card2array(card2num(message['handCards']))
        my_handcards_batch = np.repeat(my_handcards[np.newaxis, :], num_legal_actions, axis=0)

        # 万能牌的标志位, 12维
        universal_card_flag = self.proc_universal(my_handcards, RANK[message['curRank']])
        universal_card_flag_batch = np.repeat(universal_card_flag[np.newaxis, :], num_legal_actions, axis=0)

        # 其余所有玩家手上剩余的牌（根据出牌记录推断），54维
        other_hands = []
        for i in range(54): 
            if self.other_left_hands[i] == 1:
                other_hands.append(i)
            elif self.other_left_hands[i] == 2:
                other_hands.append(i)
                other_hands.append(i)
        other_handcards = card2array(other_hands)      
        other_handcards_batch = np.repeat(other_handcards[np.newaxis, :], num_legal_actions, axis=0)

        # 最新的动作，54维
        if len(self.action_seq) > 0:
            last_action = card2array(self.action_seq[-1])
        else:
            last_action = card2array([-1])
        last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)
        
        # 队友最后的动作，54维
        teammate_pos = (self.mypos + 2) % 4
        if len(self.history_action[teammate_pos]) > 0 and teammate_pos not in self.over:
            last_teammate_action = card2array(self.history_action[teammate_pos][-1])
        else:
            last_teammate_action = card2array([-1])
        last_teammate_action_batch = np.repeat(last_teammate_action[np.newaxis, :], num_legal_actions, axis=0)

        # 下家打过的牌，54维
        down_pos = (self.mypos + 1) % 4
        if len(self.history_action[down_pos]) > 0:
            down_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[down_pos]))
        else:
            down_played_cards = card2array([])
        down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        # 对家打过的牌，54维
        if len(self.history_action[teammate_pos]) > 0:
            teammate_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[teammate_pos]))
        else:
            teammate_played_cards = card2array([])
        teammate_played_cards_batch = np.repeat(teammate_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        # 上家打过的牌，54维
        up_pos = (self.mypos + 3) % 4
        if len(self.history_action[up_pos]) > 0:
            up_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[up_pos]))
        else:
            up_played_cards = card2array([])
        up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        # 下家剩余的牌数，28维
        down_num_cards_left = _get_one_hot_array(self.remaining[down_pos], 27, 1)
        down_num_cards_left_batch = np.repeat(down_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        # 对家剩余的牌数，28维
        teammate_num_cards_left = _get_one_hot_array(self.remaining[teammate_pos], 27, 1)
        teammate_num_cards_left_batch = np.repeat(teammate_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        # 上家剩余的牌数，28维
        up_num_cards_left = _get_one_hot_array(self.remaining[up_pos], 27, 1)
        up_num_cards_left_batch = np.repeat(up_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        # 己方当前的级牌，13维
        self_rank = _get_one_hot_array(RANK[message['selfRank']], 13, 0)
        self_rank_batch = np.repeat(self_rank[np.newaxis, :], num_legal_actions, axis=0)

        # 敌方当前的级牌，13维
        oppo_rank = _get_one_hot_array(RANK[message['oppoRank']], 13, 0)
        oppo_rank_batch = np.repeat(oppo_rank[np.newaxis, :], num_legal_actions, axis=0)

        # 当前的级牌，13维
        cur_rank = _get_one_hot_array(RANK[message['curRank']], 13, 0)
        cur_rank_batch = np.repeat(cur_rank[np.newaxis, :], num_legal_actions, axis=0)

        # ============ Oracle 部分: 其他三个玩家的实际手牌 ============
        # 下家手牌，54维
        if self.all_hand_cards and len(self.all_hand_cards) > down_pos and self.all_hand_cards[down_pos]:
            down_handcards = card2array(card2num(self.all_hand_cards[down_pos]))
        else:
            down_handcards = np.zeros(54)
        down_handcards_batch = np.repeat(down_handcards[np.newaxis, :], num_legal_actions, axis=0)

        # 队友手牌，54维
        if self.all_hand_cards and len(self.all_hand_cards) > teammate_pos and self.all_hand_cards[teammate_pos]:
            teammate_handcards = card2array(card2num(self.all_hand_cards[teammate_pos]))
        else:
            teammate_handcards = np.zeros(54)
        teammate_handcards_batch = np.repeat(teammate_handcards[np.newaxis, :], num_legal_actions, axis=0)

        # 上家手牌，54维
        if self.all_hand_cards and len(self.all_hand_cards) > up_pos and self.all_hand_cards[up_pos]:
            up_handcards = card2array(card2num(self.all_hand_cards[up_pos]))
        else:
            up_handcards = np.zeros(54)
        up_handcards_batch = np.repeat(up_handcards[np.newaxis, :], num_legal_actions, axis=0)

        # 合法动作，54维
        my_action_batch = np.zeros(my_handcards_batch.shape)
        for j, action in enumerate(legal_actions):
            my_action_batch[j, :] = card2array(action)

        # 组合所有特征: 513 + 162 + 54 = 729
        x_batch = np.hstack((
            # 标准部分 (513维)
            my_handcards_batch,           # 54
            universal_card_flag_batch,    # 12
            other_handcards_batch,        # 54
            last_action_batch,            # 54
            last_teammate_action_batch,   # 54
            down_played_cards_batch,      # 54
            teammate_played_cards_batch,  # 54
            up_played_cards_batch,        # 54
            down_num_cards_left_batch,    # 28
            teammate_num_cards_left_batch,# 28
            up_num_cards_left_batch,      # 28
            self_rank_batch,              # 13
            oppo_rank_batch,              # 13
            cur_rank_batch,               # 13
            # Oracle 部分 (162维)
            down_handcards_batch,         # 54
            teammate_handcards_batch,     # 54
            up_handcards_batch,           # 54
            # 动作部分 (54维)
            my_action_batch               # 54
        ))
        
        x_no_action = np.hstack((
            my_handcards,
            universal_card_flag,
            other_handcards,
            last_action,
            last_teammate_action,
            down_played_cards,
            teammate_played_cards,
            up_played_cards,
            down_num_cards_left,
            teammate_num_cards_left,
            up_num_cards_left,
            self_rank,
            oppo_rank,
            cur_rank,
            down_handcards,
            teammate_handcards,
            up_handcards
        ))
        
        obs = {
            'x_batch': x_batch.astype(np.float32),
            'legal_actions': legal_actions,
            'x_no_action': x_no_action.astype(np.float32),
        }
        return obs


def run_client(args):
    """运行单个Oracle客户端"""
    server_port = args.server_port
    position = args.position
    
    url = f'ws://127.0.0.1:{server_port}/game/client{position}'
    print(f"[Pos {position}] Connecting to {url}")
    
    client = OracleEvalClient(url, args)
    client.connect()
    client.run_forever()


def main():
    args, _ = parser.parse_known_args()
    run_client(args)


if __name__ == '__main__':
    main()
