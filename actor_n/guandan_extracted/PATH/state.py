# -*- coding: utf-8 -*-
# @Time       : 2020/10/5 15:04
# @Author     : time not yet
# @File       : state.py
# @Description:

from utils import RANK
from action import Action


class State(object):

    def __init__(self):
        self.deck_shuffle_times = 0
        self.current_pos = -1
        self.current_action = Action()
        self.greater_pos = -1
        self.greater_action = Action()

        # 用于比较单张、对子、三张、炸弹、三带二牌型的映射
        self.rank_order = {
            '2': 15, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13,
            'A': 14, 'B': 16, 'R': 17
        }

        # 用于比较钢板、连对、同花顺、顺子牌型的映射
        self.number_order = {
            'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13,
            'B': 16, 'R': 17
        }

    def action_info(self):
        return self.current_pos, self.current_action.to_json(), self.greater_pos, self.greater_action.to_json()

    def update_greater_action(self, action):
        self.greater_pos = self.current_pos
        self.greater_action = action

    def update_order(self, old_rank, new_rank):
        self.rank_order[old_rank] = RANK.index(old_rank)
        self.rank_order[new_rank] = 15

    def clear_action(self):
        self.current_pos = -1
        self.current_action.clear()
        self.greater_pos = -1
        self.greater_action.clear()

    def reset(self):
        self.deck_shuffle_times = 0
        self.clear_action()

