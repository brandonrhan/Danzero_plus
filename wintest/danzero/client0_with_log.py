#!/usr/bin/env python
"""
掼蛋训练数据生成器
使用 Danzero 神经网络打掼蛋并记录日志，用于 LLM SFT 训练

Usage:
    cd /mnt/c/Users/admin/Documents/GitHub/Danzero_plus/wintest/danzero
    source ~/miniconda/bin/activate tf115
    python client0_with_log.py --log_dir /path/to/logs --position 0
"""

import json
import os
import warnings
from argparse import ArgumentParser
from functools import reduce
from random import randint
import time
from datetime import datetime

import numpy as np
import zmq
from pyarrow import deserialize, serialize
from util import card2array, card2num, combine_handcards, num2card
from ws4py.client.threadedclient import WebSocketClient

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = ArgumentParser()
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--action_port', type=int, default=6000)
parser.add_argument('--log_dir', type=str, default='./logs_sft',
                    help='Directory to save SFT training logs')
parser.add_argument('--position', type=int, default=0, choices=[0, 2],
                    help='Player position (0 or 2)')

RANK = {
    '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8,
    'T':9, 'J':10, 'Q':11, 'K':12, 'A':13
}

def _get_one_hot_array(num_left_cards, max_num_cards, flag):
    # 添加边界检查
    if num_left_cards < 0:
        num_left_cards = 0
    if num_left_cards > max_num_cards:
        num_left_cards = max_num_cards
    if flag == 0:
        one_hot = np.zeros(max_num_cards)
        if num_left_cards > 0:
            one_hot[num_left_cards - 1] = 1
    else:
        one_hot = np.zeros(max_num_cards+1)
        one_hot[num_left_cards] = 1
    return one_hot

def _action_seq_list2array(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = card2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 216)
    return action_seq_array

def _process_action_seq(sequence, length=20):
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


def getlist(handcards, rank):
    """获取所有合法动作列表"""
    single_actionlist = []
    pair_actionlist = []
    trips_actionlist = []
    threepair_actionlist = []
    threetwo_actionlist = []
    twotrips_actionlist = []
    straight_actionlist = []

    rank_card = 'H' + str(rank)
    card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                        "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}
    card_value_s2v2 = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                        "Q": 12, "K": 13, "B": 16, "R": 17}
    card_value_s2v[rank_card[-1]] = 15
    sorted_cards, bomb_info = combine_handcards(handcards, rank, card_value_s2v)
    
    def mysort(elem):
        return card_value_s2v[elem[1]]

    def mysort1(elem):
        return card_value_s2v2[elem[1]]

    if sorted_cards["Single"]:
        for singlecard in sorted_cards['Single']:
            single_actionlist.append(['Single', singlecard[-1], [singlecard]])
        single_actionlist.sort(key=mysort)

    if sorted_cards["Pair"]:
        for paircard in sorted_cards['Pair']:
            pair_actionlist.append(['Pair', paircard[0][-1], paircard])
        pair_actionlist.sort(key=mysort)

    if sorted_cards['Trips']:
        for tripcard in sorted_cards['Trips']:
            trips_actionlist.append(['Trips', tripcard[0][-1], tripcard])
        trips_actionlist.sort(key=mysort)

    if sorted_cards['Pair'] and sorted_cards['Trips']:
        for tripcard in sorted_cards['Trips']:
            for paircard in sorted_cards['Pair']:
                threetwo_actionlist.append(['ThreeWithTwo', tripcard[0][-1], tripcard + paircard])
        threetwo_actionlist.sort(key=mysort)

    if len(sorted_cards['Pair']) >= 3:
        for i in range(len(pair_actionlist) - 2):
            if card_value_s2v[pair_actionlist[i][1]] == card_value_s2v[pair_actionlist[i + 1][1]] - 1 and \
                    card_value_s2v[pair_actionlist[i + 1][1]] == card_value_s2v[pair_actionlist[i + 2][1]] - 1:
                action2 = pair_actionlist[i][-1] + pair_actionlist[i + 1][-1] + pair_actionlist[i + 2][-1]
                threepair_actionlist.append(['ThreePair', action2[0][-1], action2])
        threepair_actionlist.sort(key=mysort1)

    if len(sorted_cards['Trips']) >= 2:
        for i in range(len(trips_actionlist) - 1):
            if card_value_s2v[trips_actionlist[i][1]] == card_value_s2v[trips_actionlist[i + 1][1]] - 1:
                action3 = trips_actionlist[i][-1] + trips_actionlist[i + 1][-1]
                twotrips_actionlist.append(['TwoTrips', action3[0][-1], action3])
        twotrips_actionlist.sort(key=mysort1)

    if 'Straight' in sorted_cards.keys() and sorted_cards['Straight']:
        for straightcard in sorted_cards['Straight']:
            straight_actionlist.append(['Straight', straightcard[0][-1], straightcard])
        straight_actionlist.sort(key=mysort1)

    return single_actionlist + pair_actionlist + trips_actionlist + threepair_actionlist + threetwo_actionlist + twotrips_actionlist + straight_actionlist


class SFTLoggingClient(WebSocketClient):
    """支持SFT日志记录的掼蛋客户端"""
    
    def __init__(self, url, args):
        super().__init__(url)
        self.args = args
        self.mypos = 0
        self.history_action = {0: [], 1: [], 2: [], 3:[]}
        self.action_seq = []
        self.action_order = []
        self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self.other_left_hands = [2 for _ in range(54)]
        self.flag = 0
        self.over = []
        self.rank = {'self_rank': 1, 'oppo_rank': 1}
        self.tongji = {3: 0, 2: 0, 1: 0, -1: 0, -2: 0, -3: 0, 'all': 0}
        
        # 当前游戏的 obs 记录
        self.game_obs_list = []
        self.cur_rank = '2'
        self.self_rank = '2'
        self.oppo_rank = '2'

        # 初始化zmq
        self.context = zmq.Context()
        self.context.linger = 0 
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'tcp://localhost:{6000 + args.position}')

        # 初始化日志
        print('Start evaluation at:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.start_time = time.time()
        time_str = time.strftime('-%Y%m%d%H%M%S', time.localtime(self.start_time))
        
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        self.log_path = os.path.join(args.log_dir, f'log-client{args.position}{time_str}.jsonl')
        print(f'Logging to: {self.log_path}')

    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def log_data(self, data):
        """记录一条数据到日志文件"""
        if self.log_path:
            print(f'[DEBUG] Writing to {self.log_path}')
            with open(self.log_path, 'a', encoding="utf-8") as fout:
                fout.write(json.dumps(data, ensure_ascii=False))
                fout.write('\n')
                fout.flush()  # 确保立即写入

    def build_obs_for_sft(self, message, action):
        """构建用于SFT训练的观察数据"""
        obs = {
            'my_pos': self.mypos,
            'my_hands': message['handCards'],
            'remaining_hands': sum(self.remaining.values()) - self.remaining[self.mypos],
            'last_action': message['actionList'][0] if len(self.action_seq) == 0 or self.action_seq[-1] == [] else 
                          ['PASS', 'PASS', 'PASS'] if self.action_seq[-1] == [-1] else 
                          self._num2action(self.action_seq[-1]) if self.action_seq[-1] else ['PASS', 'PASS', 'PASS'],
            'last_teammate_action': ['PASS', 'PASS', 'PASS'],  # 简化处理
            'number_of_cards_left': [self.remaining[i] for i in range(4)],
            'down_played_cards': self._get_played_cards((self.mypos + 1) % 4),
            'teammate_played_cards': self._get_played_cards((self.mypos + 2) % 4),
            'up_played_cards': self._get_played_cards((self.mypos + 3) % 4),
            'self_rank': self.self_rank,
            'oppo_rank': self.oppo_rank,
            'cur_rank': self.cur_rank,
            'legal_actions': message['actionList']
        }
        return {'obs': obs, 'action': action}

    def _num2action(self, nums):
        """将数字列表转换为动作格式（简化版）"""
        if not nums or nums == [-1]:
            return ['PASS', 'PASS', 'PASS']
        cards = [num2card(n) for n in nums]
        # 简单返回牌面
        return ['Cards', str(len(cards)), cards]

    def _get_played_cards(self, pos):
        """获取某位置打过的牌"""
        result = []
        for action in self.history_action[pos]:
            if action and action != [-1]:
                result.append(self._num2action(action))
        return result

    def proc_universal(self, my_handcards, rank):
        universal_card_flag = np.zeros(12)
        for idx, i in enumerate([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]):
            if rank == idx + 2:
                if my_handcards[52] > 0:
                    universal_card_flag[idx] = 1
            else:
                if my_handcards[i + 1] > 0 and my_handcards[i + 2] > 0:
                    universal_card_flag[idx] = 2
                elif my_handcards[i + 1] > 0 or my_handcards[i + 2] > 0:
                    universal_card_flag[idx] = 1
        return universal_card_flag

    def received_message(self, message):
        message = json.loads(str(message))
        if message['type'] == 'notify':
            if message['stage'] == 'beginning':
                self.mypos = message['myPos']
                self.cur_rank = message['curRank']
                self.self_rank = message['selfRank']
                self.oppo_rank = message['oppoRank']
            elif message['stage'] == 'tribute':
                self.tribute_result = message['result']
            elif message['stage'] == 'play':
                just_play = message['curPos']
                action = card2num(message['curAction'][2])
                if message['curPos'] != self.mypos:
                    for ele in action:
                        self.other_left_hands[ele] -= 1
                if len(self.over) == 0:
                    self.action_order.append(just_play)
                    self.action_seq.append(action)
                    self.history_action[message['curPos']].append(action)
                elif len(self.over) == 1:
                    if len(action) > 0 and self.flag == 1:
                        self.flag = 2
                    if self.flag <= 2 and (just_play+1) % 4 not in self.over:
                        self.action_order.append(just_play)
                        self.action_seq.append(action)
                        self.history_action[message['curPos']].append(action)
                    elif self.flag <= 2 and (just_play+2) % 4 not in self.over:
                        self.action_order.append(just_play)
                        self.action_seq.append(action)
                        self.history_action[message['curPos']].append(action)
                        self.action_order.append((just_play + 1) % 4)
                        self.history_action[(just_play + 1) % 4].append([-1])
                        self.action_seq.append([-1])
                    elif self.flag <= 2 and (just_play+3) % 4 not in self.over:
                        self.flag = 3
                        self.action_order.append(just_play)
                        self.action_seq.append(action)
                        self.history_action[message['curPos']].append(action)
                        self.action_order.append((just_play + 1) % 4)
                        self.history_action[(just_play + 1) % 4].append([-1])
                        self.action_seq.append([-1])
                        self.action_order.append((just_play + 2) % 4)
                        self.history_action[(just_play + 2) % 4].append([-1])
                        self.action_seq.append([-1])
                    elif self.flag <= 2 and (just_play+1) % 4 in self.over:
                        self.flag = 3
                        self.action_order.append(just_play)
                        self.action_seq.append(action)
                        self.history_action[message['curPos']].append(action)
                        self.action_order.append((just_play + 1) % 4)
                        self.history_action[(just_play + 1) % 4].append([-1])
                        self.action_seq.append([-1])
                        self.action_order.append((just_play + 2) % 4)
                        self.history_action[(just_play + 2) % 4].append([-1])
                        self.action_seq.append([-1])
                        if just_play == (self.over[-1] + 2) % 4:
                            self.action_order.append((just_play + 3) % 4)
                            self.history_action[(just_play + 3) % 4].append([])
                            self.action_seq.append([])
                    elif (just_play+1) % 4 in self.over and self.flag == 3:
                        self.action_order.append(just_play)
                        self.action_seq.append(action)
                        self.history_action[message['curPos']].append(action)
                        self.action_order.append((just_play + 1) % 4)
                        self.history_action[(just_play + 1) % 4].append([-1])
                        self.action_seq.append([-1])
                        self.action_order.append((just_play + 2) % 4)
                        self.history_action[(just_play + 2) % 4].append([-1])
                        self.action_seq.append([-1])
                    else:
                        self.action_order.append(just_play)
                        self.action_seq.append(action)
                        self.history_action[message['curPos']].append(action)

                # 只在普通出牌时减少remaining，跳过进贡/还贡等特殊操作
                cur_action_type = message.get('curAction', [None])[0]
                if cur_action_type not in ['tribute', 'back', 'anti-tribute']:
                    self.remaining[just_play] -= len(action)
                    # 确保remaining不会变成负数
                    if self.remaining[just_play] < 0:
                        self.remaining[just_play] = 0
                    if self.remaining[just_play] == 0:
                        self.over.append(just_play)

        elif message["type"] == 'act':
            if message["stage"] == "back":
                act_index = self.back_action(message, self.mypos, self.tribute_result)
                self.send(json.dumps({"actIndex": int(act_index)}))
            elif message["stage"] == "tribute":
                act_index = self.tribute(message['actionList'], message["curRank"])
                self.send(json.dumps({"actIndex": int(act_index)}))
            elif message["stage"] == 'play':
                if self.flag == 0:
                    init_hand = card2num(message['handCards'])
                    for ele in init_hand:
                        self.other_left_hands[ele] -= 1
                    self.flag = 1

                if len(message['actionList']) == 1:
                    # 记录只有一个选择的情况
                    obs_data = self.build_obs_for_sft(message, message['actionList'][0])
                    self.game_obs_list.append(obs_data)
                    self.send(json.dumps({"actIndex": 0}))
                else:
                    state = self.prepare(message)
                    self.socket.send(serialize(state).to_buffer())
                    act_index = deserialize(self.socket.recv())
                    
                    # 记录观察和动作
                    chosen_action = message['actionList'][act_index]
                    obs_data = self.build_obs_for_sft(message, chosen_action)
                    self.game_obs_list.append(obs_data)
                    
                    self.send(json.dumps({"actIndex": int(act_index)}))

        # 小局结束，处理日志记录和数据重置
        if message['stage'] == 'episodeOver':
            # 计算奖励
            win_flag = self.get_reward(message)
            self.tongji[win_flag] += 1
            self.tongji['all'] += 1
            
            print(f'[DEBUG] Game ended: win_flag={win_flag}, obs_count={len(self.game_obs_list)}')
            
            # 只记录获胜方的数据
            if win_flag > 0:
                print(f'[DEBUG] Writing {len(self.game_obs_list)} observations to log')
                for obs_data in self.game_obs_list:
                    self.log_data(obs_data)
                # 记录奖励行
                self.log_data({'reward': win_flag})
            
            # 打印统计
            if self.tongji['all'] % 10 == 0:
                print(f"Games: {self.tongji['all']}, "
                      f"Win: {self.tongji[3] + self.tongji[2] + self.tongji[1]}, "
                      f"Lose: {self.tongji[-1] + self.tongji[-2] + self.tongji[-3]}")
            
            # 重置游戏状态
            self.history_action = {0: [], 1: [], 2: [], 3:[]}
            self.action_seq = []
            self.action_order = []
            self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
            self.other_left_hands = [2 for _ in range(54)]
            self.over = []
            self.flag = 0
            self.game_obs_list = []

    def get_reward(self, message):
        """计算奖励（复制自 client0.py）"""
        team = [self.mypos, (self.mypos + 2) % 4]
        order = message['order']
        rewards = {"1100": 3, "1010": 2, "1001": 1, "0110": -1, "0101": -2, "0011": -3}
        res = ""
        for i in order:
            if i in team:
                res += '1'
            else:
                res += '0'
        return rewards[res]

    def prepare(self, message):
        """准备神经网络输入状态"""
        num_legal_actions = message['indexRange'] + 1
        legal_actions = [card2num(i[2]) for i in message['actionList']]
        my_handcards = card2array(card2num(message['handCards']))
        my_handcards_batch = np.repeat(my_handcards[np.newaxis, :], num_legal_actions, axis=0)

        universal_card_flag = self.proc_universal(my_handcards, RANK[message['curRank']])
        universal_card_flag_batch = np.repeat(universal_card_flag[np.newaxis, :], num_legal_actions, axis=0)

        other_hands = []
        for i in range(54):
            if self.other_left_hands[i] == 1:
                other_hands.append(i)
            elif self.other_left_hands[i] == 2:
                other_hands.append(i)
                other_hands.append(i)
        other_handcards = card2array(other_hands)
        other_handcards_batch = np.repeat(other_handcards[np.newaxis, :], num_legal_actions, axis=0)

        last_action = card2array(self.action_seq[-1]) if len(self.action_seq) > 0 else card2array([-1])
        last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)

        last_teammate_action = card2array([-1])
        if len(self.history_action[(self.mypos + 2) % 4]) > 0 and (self.mypos + 2) % 4 not in self.over:
            last_teammate_action = card2array(self.history_action[(self.mypos + 2) % 4][-1])
        last_teammate_action_batch = np.repeat(last_teammate_action[np.newaxis, :], num_legal_actions, axis=0)

        my_action_batch = np.zeros(my_handcards_batch.shape)
        for j, action in enumerate(legal_actions):
            my_action_batch[j, :] = card2array(action)

        down_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 1) % 4], 27, 1)
        down_num_cards_left_batch = np.repeat(down_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        teammate_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 2) % 4], 27, 1)
        teammate_num_cards_left_batch = np.repeat(teammate_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        up_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 3) % 4], 27, 1)
        up_num_cards_left_batch = np.repeat(up_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 1) % 4]) > 0:
            down_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 1) % 4]))
        else:
            down_played_cards = card2array([])
        down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 2) % 4]) > 0:
            teammate_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 2) % 4]))
        else:
            teammate_played_cards = card2array([])
        teammate_played_cards_batch = np.repeat(teammate_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 3) % 4]) > 0:
            up_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 3) % 4]))
        else:
            up_played_cards = card2array([])
        up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        self_rank = _get_one_hot_array(RANK[message['selfRank']], 13, 0)
        self_rank_batch = np.repeat(self_rank[np.newaxis, :], num_legal_actions, axis=0)

        oppo_rank = _get_one_hot_array(RANK[message['oppoRank']], 13, 0)
        oppo_rank_batch = np.repeat(oppo_rank[np.newaxis, :], num_legal_actions, axis=0)

        cur_rank = _get_one_hot_array(RANK[message['curRank']], 13, 0)
        cur_rank_batch = np.repeat(cur_rank[np.newaxis, :], num_legal_actions, axis=0)

        x_batch = np.hstack((my_handcards_batch,
                        universal_card_flag_batch,
                        other_handcards_batch,
                        last_action_batch,
                        last_teammate_action_batch,
                        down_played_cards_batch,
                        teammate_played_cards_batch,
                        up_played_cards_batch,
                        down_num_cards_left_batch,
                        teammate_num_cards_left_batch,
                        up_num_cards_left_batch,
                        self_rank_batch,
                        oppo_rank_batch,
                        cur_rank_batch,
                        my_action_batch))

        state = dict()
        state['x_batch'] = x_batch
        state['legal_actions'] = legal_actions
        return state

    def tribute(self, actionList, curRank):
        """进贡策略"""
        card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                        "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}
        card_value_s2v[curRank] = 15
        tribute_card = ['', -1]
        if len(actionList) == 1:
            return 0
        for i, action in enumerate(actionList):
            if action[2][0][1] == 'R' or action[2][0][1] == 'B':
                return i
        for i, action in enumerate(actionList):
            if card_value_s2v[action[2][0][1]] > tribute_card[1]:
                tribute_card = [action[2], card_value_s2v[action[2][0][1]], i]
        return tribute_card[2]

    def back_action(self, message, mypos, tribute_result):
        """还贡策略"""
        # tribute_result 可能是不同格式，需要健壮处理
        if tribute_result:
            try:
                for i, result in enumerate(tribute_result):
                    # 检查 result 是否是字典格式
                    if isinstance(result, dict) and 'to' in result:
                        if result['to'] == mypos:
                            if result['card'][1] == 'R' or result['card'][1] == 'B':
                                return 0
            except (KeyError, TypeError, IndexError):
                pass  # 忽略格式错误，使用默认还贡策略
        
        card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                        "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}
        card_value_s2v[message['curRank']] = 15
        back_card = ['', 100, 0]
        for i, action in enumerate(message['actionList']):
            if card_value_s2v[action[2][0][1]] < back_card[1]:
                back_card = [action[2], card_value_s2v[action[2][0][1]], i]
        return back_card[2]


if __name__ == '__main__':
    args = parser.parse_args()
    # 正确的URL格式：使用固定端口23456，通过路径区分client
    url = f"ws://127.0.0.1:23456/game/client{args.position}"
    ws = SFTLoggingClient(url, args)
    ws.connect()
    ws.run_forever()
