# -*- coding: utf-8 -*-
# @Time       : 2020/10/4 14:08
# @Author     : time not yet
# @File       : player.py
# @Description: player

from collections import Counter

from card import Card
from utils import CARD_DIGITAL_TABLE, cmp2cards, CARD_TABLE


class Player(object):

    def __init__(self, name, index):
        """
        player包含如下属性：
        name:        名称
        pos:         座位号
        victory:     胜利次数
        hearts_num:  红桃配的数量
        hand_cards:  手牌
        play_area:   出牌区域
        rank:        等级
        stuck_times: 停留在A的次数
        :param name: 名称
        """
        self.name = name
        self.pos = index
        self.victory = 0
        self.hearts_num = 0
        self.hand_cards = []
        self.play_area = None
        self.rank = 2
        self.stuck_times = 0

    def hand2json(self):
        return [str(card) for card in self.hand_cards]

    def hand2str(self):
        return str([CARD_TABLE[str(card)] for card in self.hand_cards])

    def update_play_area(self, play_area):
        self.play_area = play_area

    def update_rank(self, inc):
        self.rank += inc
        self.rank = min(self.rank, 14)

    def public_info(self):
        """
        返回player的公共信息: 剩余手牌数量、出牌区域
        :return: dict
        """
        return {"rest": len(self.hand_cards), "playArea": self.play_area.to_json() if self.play_area else None}

    def add_rank(self, inc):
        """
        增加player的等级，最大等级不超过14
        :param inc: 等级增量
        :return: None
        """
        self.rank += inc
        self.rank = min(self.rank, 14)

    def reset(self, **kwargs):
        """
        根据关键字参数重置player的属性
        :param kwargs: 关键字参数表示的属性
        :return: None
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def play_card(self, cards, hearts_card):
        """
        执行player的打牌动作。当打出红桃配时，需要减去对应的红桃配的数量。
        :param cards: 需要打出的牌的列表
        :param hearts_card: 当前等级
        :return: None
        """
        for card in cards:
            self.hand_cards.remove(card)
            if card == hearts_card:
                self.hearts_num -= 1

    def add_card(self, card, current_rank):
        """
        将卡牌添加到玩家手中，并保持牌序有序
        :param card: 添加的卡牌
        :param current_rank: 当前等级
        :return: None
        """
        index = 0
        temp_card = Card(card[0], card[1], digital=CARD_DIGITAL_TABLE[card])
        while index != len(self.hand_cards):
            if cmp2cards(temp_card, self.hand_cards[index], current_rank, False):
                self.hand_cards.insert(index, temp_card)
                break
            index += 1
        if index == len(self.hand_cards):
            self.hand_cards.append(temp_card)

    @property
    def red_joker_num(self):
        """
        返回player手中大王的数量。由于手牌的是点数小到大、花色从黑桃到方片进行排序，所以若有大王，则大王必在手牌的倒数一、二张的位置。
        :return: int
        """
        if self.hand_cards[-1] == "HR":
            if self.hand_cards[-2] == "HR":
                return 2
            else:
                return 1
        else:
            return 0

    def max_cards(self, rank, rank_card):
        """
        返回player手中最大的牌（不含红桃配）
        0.由于手牌是从小到大、花色是从黑桃到方片进行排序，所以若最后一张为大王或小王，则直接返回大王或者小王。
        1.若最后一张是红桃配，且红桃配前面仍有和当前rank一样的牌，则在进贡时需要避免进贡红桃配。（如[..., 黑桃2, 红桃2, 红桃2]）
        2.若最后一张牌的rank等于当前的rank，则在进贡时需要避免进贡红桃配。（如[..., 黑桃2, 红桃2, 红桃2, 方片2]）
        若满足条件1，则player手中除红桃配外最大的牌即为跳过红桃配之后的那张牌，该牌的位置用负索引表示
        若满足条件2，则player手中最大的牌的位置所对应的负索引即为-1，且需要避开红桃花色
        :param rank: 当前等级
        :param rank_card: 红桃配
        :return: list
        """
        if self.hand_cards[-1] == "HR":
            return [["tribute", "tribute", ["HR"]]]
        elif self.hand_cards[-1] == "SB":
            return [["tribute", "tribute", ["SB"]]]
        else:
            avoid_hearts = False
            index = -1
            if self.hand_cards[-1] == rank_card:
                if self.hand_cards[-1 - self.hearts_num].rank == rank:
                    avoid_hearts = True
                index = -1 - self.hearts_num
            elif self.hand_cards[-1].rank == rank and self.hand_cards[-1] != rank_card and self.hearts_num:
                avoid_hearts = True
            if avoid_hearts:
                cards = [
                    str(card) for card in self.hand_cards
                    if card.rank == self.hand_cards[index].rank and card.suit != 'H'
                ]
            else:
                cards = [str(card) for card in self.hand_cards if card.rank == self.hand_cards[index].rank]
            max_cards = []
            for key in Counter(cards).keys():
                max_cards.append(["tribute", "tribute", [key]])
            return max_cards

    def less_than_ten(self, current_rank):
        """
        返回player手中小于10的牌。
        :return:
        """
        # return list(
        #     Counter(
        #         [str(card) for card in self.hand_cards if card.digital & 0x00FF <= 10 and card.rank != current_rank]
        #     ).keys()
        # )
        cards = [str(card) for card in self.hand_cards if card.digital & 0x00FF <= 10 and card.rank != current_rank]
        less_than_ten = []
        for key in Counter(cards).keys():
            less_than_ten.append(["back", "back", [key]])
        return less_than_ten

