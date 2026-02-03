# -*- coding: utf-8 -*-
# @Time       : 2020/10/4 14:05
# @Author     : time not yet
# @File       : env.py
# @Description: environment

import logging
from copy import deepcopy
from random import shuffle, randint
from collections import namedtuple

from card import Card
from state import State
from player import Player
from action import ActionList, Action
from utils import cmp2cards, cmp2rank, OverOrder, JSONGen

MsgToBeSend = namedtuple("MsgToBeSend", ['pos', 'msg'])
STAGE_PLAY = "play"
STAGE_TRIBUTE = "tribute"
STAGE_BACK = "back"


def win_at_ace(over_order, rank_belongs, first_rank, cur_rank):
    if (over_order[0] + 2) % 4 == over_order[1] or (over_order[0] + 2) % 4 == over_order[2]:
        if (rank_belongs + 2) % 4 == over_order[0] and first_rank == cur_rank == 14:
            return True
        else:
            return False
    return False


class Environment(object):

    def __init__(self, times):
        """
        根据掼蛋游戏规则，模拟掼蛋游戏环境。新建实例时，初始化108张有序卡牌
        实例属性说明
        私有属性：
            cards：108张有序卡牌
            rank：当前所打的等级，以数字的形式表示
        公有属性：
            times：当前游戏次数
            total：设定的游戏次数
            deck：牌堆，需要洗牌时直接将__cards深复制给deck
            players：玩家列表
            rank_belongs：当前等级归属于哪个队伍
            state：状态类，负责保存当前动作，最大动作
            action：动作类，负责生成动作，给出动作索引范围
            over_order：辅助类，负责记录完牌次序，结算进贡还贡关系
            loop：该属性用于实现游戏中各个阶段的状态跳转
        :param times: 游戏局数
        """
        self.__cards = []
        self.__rank = 2
        self.times = 0
        self.total = times
        self.shuffle_times = 0
        self.deck = None
        self.players = []
        self.rank_belongs = -1
        self.state = State()
        self.action = ActionList()
        self.over_order = OverOrder()
        self.results = None
        self.record_file = None
        self.loop = self.start

        for i in range(2):
            digital_form = 0x0101
            for s in Card.ALL_SUITS:
                for r in Card.ALL_RANKS[2:]:
                    digital_form += 1
                    self.__cards.append(Card(s, r, digital_form))
                digital_form = digital_form & 0x0F01
                digital_form += 0x0101
            self.__cards.append(Card('S', 'B', 0x0110))  # 小王
            self.__cards.append(Card('H', 'R', 0x0211))  # 大王

    @property
    def rank(self):
        """
        以字符串的形式返回当前所打的等级
        :return: str
        """
        return Card.ALL_RANKS[self.__rank]

    def set_rank(self, rank, belongs):
        """
        设置当前等级和当前等级归属于哪个队伍
        :param rank: 当前等级
        :param belongs: 等级归属
        :return:
        """
        self.__rank = rank
        self.rank_belongs = belongs

    def add_victory_num(self, pos, teammate_pos):
        """
        为胜利队伍的队员增加胜利场次
        :param pos: 首先完牌的玩家的座位号
        :param teammate_pos: 其队友的座位号
        """
        self.players[pos].victory += 1
        self.players[teammate_pos].victory += 1

    def first_action_right_transfer(self):
        """
        率先出牌的牌权是否发生转移。满足以下两个条件，则牌权发生转移：
        （1）
            1.当前动作的玩家的下家是当前最大动作的玩家或当前动作的玩家的下下家是当前最大动作的玩家
            2.最大动作的玩家已经完牌
        （2）
            1.当前动作的玩家的下下家是当前最大动作的玩家
            2.最大动作的玩家已经完牌，当前动作的玩家的下家玩家已经完牌
        局部变量含义说明：
            c_pos：当前动作的玩家的座位号
            g_pos：当前最大动作的玩家的座位号
        :return: True or False
        """
        c_pos, g_pos = self.state.current_pos, self.state.greater_pos
        prerequisite_a = (c_pos + 1) % 4 == g_pos  # or (c_pos + 2) % 4 == g_pos
        prerequisite_b = len(self.players[self.state.greater_pos].hand_cards) == 0

        prerequisite_A = (c_pos + 2) % 4 == g_pos
        prerequisite_B = len(self.players[self.state.greater_pos].hand_cards) == 0 \
                         and len(self.players[(c_pos + 1) % 4].hand_cards) == 0

        # print((prerequisite_a and prerequisite_b), (prerequisite_A and prerequisite_B))
        return (prerequisite_a and prerequisite_b) or (prerequisite_A and prerequisite_B)

    def enter_tribute_stage(self, first_pos, rank_inc, msg_to_be_send):
        """
        准备进入进贡阶段，更新头游队伍等级，更新当前所打的队伍，进入进贡阶段进行结算
        :param first_pos: 第一名座位号
        :param rank_inc: 等级增量
        :param msg_to_be_send: 待发送消息列表
        :return:
        """
        self.players[first_pos].update_rank(rank_inc)
        self.players[(first_pos + 2) % 4].update_rank(rank_inc)
        self.state.update_order(old_rank=self.rank, new_rank=Card.ALL_RANKS[self.players[first_pos].rank])
        self.set_rank(self.players[first_pos].rank, belongs=first_pos)
        for player in self.players:
            player.reset(play_area=None, hearts_num=0)
        self.reshuffle()
        self.deal()
        self.loop = self.tribute
        self.loop(msg_to_be_send=msg_to_be_send)

    def next_player_second_action(self, msg_to_be_send):
        """
        下一位有牌的玩家做后手动作
        :param msg_to_be_send: 待发送消息列表
        """
        current_pos = self.next_have_cards_pos()
        self.second_action(current_pos)
        self.act(msg_to_be_send, current_pos, STAGE_PLAY)
        self.state.current_pos = current_pos

    def stuck_at_ace(self, first_pos, teammate_pos, rank_inc, msg_to_be_send):
        """
        当前等级为第一名的队伍所在的A等级，但第一名的队友取得了第四名的成绩，故停留在A
        :param first_pos: 第一位完牌的玩家的座位号
        :param teammate_pos: 第一位完牌的玩家的队友的座位号
        :param rank_inc: 等级增量
        :param msg_to_be_send: 待发送消息列表
        """

        self.players[first_pos].stuck_times += 1
        self.players[teammate_pos].stuck_times += 1
        if self.players[first_pos].stuck_times > 3:
            self.shuffle_times += 1
            if self.shuffle_times >= 50:
                self.times += 1
                self.notify_game_over(msg_to_be_send)
                winner, teammate = self.shuffle_times_exceeds_threshold()
                self.add_victory_num(winner, teammate)
                if self.times >= self.total:
                    self.notify_game_result(msg_to_be_send)
                else:
                    self.new_times(old_rank='2', new_rank='2', first=winner, msg_to_be_send=msg_to_be_send)
            else:
                self.players[first_pos].rank = 2
                self.players[(first_pos + 2) % 4].rank = 2
                self.players[first_pos].stuck_times = 0
                self.players[(first_pos + 2) % 4].stuck_times = 0
                for player in self.players:
                    player.reset(play_area=None)
                self.state.update_order(old_rank=self.rank, new_rank='2')
                self.set_rank(self.players[first_pos].rank, belongs=first_pos)
                self.reshuffle()
                self.deal()
                self.loop = self.tribute
                self.loop(msg_to_be_send=msg_to_be_send)
        else:
            self.enter_tribute_stage(first_pos, rank_inc, msg_to_be_send)

    def one_times_over(self, first_pos, msg_to_be_send):
        self.times += 1
        self.add_victory_num(first_pos, (first_pos + 2) % 4)
        if self.times >= self.total:
            self.notify_game_over(msg_to_be_send)
            self.notify_game_result(msg_to_be_send)
        else:
            self.new_times(old_rank='2', new_rank='2', first=first_pos, msg_to_be_send=msg_to_be_send)

    def shuffle_times_exceeds_threshold(self):
        """
        当双方从A降级到2的次数总和超过阈值50次之后，强行进行结算，不管谁赢，都将进行下一次新的游戏。比较等级只是为了统计哪一方增加胜场
        :return:
        """
        if self.players[0].rank > self.players[1].rank:
            return 0, 2
        elif self.players[0].rank < self.players[1].rank:
            return 1, 3
        else:
            if self.rank_belongs in (0, 2):
                return 0, 2
            else:
                return 1, 3

    def new_times(self, old_rank, new_rank, first, msg_to_be_send):
        """
        某一方处于A等级并且满足胜利条件，或者强行重新开局时，调用此函数。首先清空所有玩家的出牌区域，所有玩家停留A的次数，重置所有等级为2。
        然后重新洗牌，发牌，接着执行开始阶段（开始阶段会将over_order重置，状态重置），最后进行状态跳转
        :param old_rank:旧等级(str)
        :param new_rank:新等级
        :param first:头游
        :param msg_to_be_send:待发送消息列表
        :return:
        """
        for player in self.players:
            player.reset(play_area=None, stuck_times=0, rank=2, hearts_num=0)
        self.state.update_order(old_rank, new_rank)
        self.set_rank(rank=2, belongs=first)
        self.reshuffle()
        self.deal()
        msg_to_be_send.extend(self.start())
        self.loop = self.play

    def reshuffle(self):
        """
        重新洗牌
        """
        self.deck = deepcopy(self.__cards)
        shuffle(self.deck)

    def deal(self):
        """
        分发卡牌给四位玩家，并按照点数从小到大，花色从黑桃、红桃、梅花、方片的顺序进行排序（使用插入排序），并记录每位玩家手中的红桃配的数量
        局部变量说明：
            count：牌堆的卡牌索引
            index：某张卡牌插入手牌的位置
        """
        count = 107
        for i in range(4):
            if len(self.players[i].hand_cards) != 0:
                self.players[i].hand_cards = []
            self.players[i].hearts_num = 0
            for j in range(27):
                if self.deck[count].rank == self.rank and self.deck[count].suit == 'H':
                    self.players[i].hearts_num += 1

                # 边发牌边进行手牌排序，根据当前所打等级从小到大排列，同点数则按照黑桃、红桃、梅花、方片的顺序排列
                index = 0
                while index != len(self.players[i].hand_cards):
                    if cmp2cards(self.deck[count], self.players[i].hand_cards[index], self.rank, False):
                        self.players[i].hand_cards.insert(index, self.deck[count])
                        break
                    index += 1
                if index == len(self.players[i].hand_cards):
                    self.players[i].hand_cards.append(self.deck[count])
                count -= 1
                self.deck.pop()

    def validate(self, pos, action):
        """
        验证玩家发来的内容是否正确。验证玩家座位号是否是当前需要做出动作的玩家，验证发送的动作索引是否在合法范围内
        :param pos: 玩家座位号
        :param action: 玩家所发送来的动作
        :return: True or False
        """
        if pos == self.state.current_pos and "actIndex" in action and \
                action["actIndex"] in self.action.valid_range:
            return True
        else:
            return False

    def next_have_cards_pos(self):
        """
        获得下一位有牌的玩家的座位号
        """
        for i in range(1, 4):
            if len(self.players[(self.state.current_pos + i) % 4].hand_cards) > 0:
                print((self.state.current_pos + i) % 4)
                return (self.state.current_pos + i) % 4

    def notify_beginning(self, msg_list):
        """
        游戏开始阶段，通知每位玩家各自的手牌情况
        :param msg_list: 待发送消息列表
        """
        # for player in self.players:
        #     msg_list.append(
        #         MsgToBeSend(
        #             player.pos,
        #             JSONGen.notify_beginning(
        #                 player.hand_cards, player.pos, self.__rank, self.players[2].rank, self.players[1].rank
        #             )
        #         )
        #     )

        for i in range(len(self.players)):
            player = self.players[i]
            opponent = self.players[(i + 1) % 4]
            msg_list.append(
                MsgToBeSend(
                    player.pos,
                    JSONGen.notify_beginning(
                        player.hand_cards, player.pos, self.__rank, player.rank, opponent.rank
                    )
                )
            )

    def notify_play(self, msg_list):
        """
        出牌阶段，用于通知其他玩家做出了什么动作
        :param msg_list: 待发送消息列表
        """
        for player in self.players:
            msg_list.append(MsgToBeSend(player.pos, JSONGen.notify_play(*self.state.action_info())))

    def notify_episode_over(self, msg_list):
        """
        小局结束阶段，用于通知所有玩家小局结束
        :param msg_list: 待发送消息列表
        """
        res = [[player.pos, player.hand2json()] for player in self.players if player.hand_cards]
        order = deepcopy(self.over_order.order)
        for player in self.players:
            msg_list.append(MsgToBeSend(player.pos, JSONGen.notify_episode_over(self.rank, order, res)))

    def notify_game_over(self, msg_list):
        """
        到达指定游戏次数游戏结束，用于通知所有玩家游戏结束
        :param msg_list: 待发送消息列表
        """
        for player in self.players:
            msg_list.append(MsgToBeSend(player.pos, JSONGen.notify_game_over(self.times, self.total)))

    def notify_game_result(self, msg_list):
        """
        通知所有玩家胜利次数、和局次数
        :param msg_list: 待发送消息列表
        局部变量说明：
            respective_wins：各个玩家的胜场次数
            respective_draws：各个玩家的和局次数
        """
        respective_wins = [player.victory for player in self.players]
        respective_draws = [0, 0, 0, 0]
        self.results = [player.victory for player in self.players]
        logging.info("{}次结束，其中0号位胜利{}次，1号位胜利{}次，2号位胜利{}次，2号位胜利{}次".format(self.total, *respective_wins))
        for player in self.players:
            msg_list.append(MsgToBeSend(player.pos, JSONGen.notify_game_result(respective_wins, respective_draws)))

    def notify_tribute(self, msg_list):
        """
        进贡阶段，用于通知所有玩家进贡者（们）都进贡了什么牌
        :param msg_list: 待发送消息列表
        """
        for player in self.players:
            msg_list.append(MsgToBeSend(player.pos, JSONGen.notify_tribute(deepcopy(self.over_order.tri_cards))))

    def notify_back(self, msg_list):
        """
        还贡阶段，用于通知所有玩家还贡者（们）都还贡了什么牌
        :param msg_list: 待发送消息列表
        """
        for player in self.players:
            msg_list.append(MsgToBeSend(player.pos, JSONGen.notify_back(deepcopy(self.over_order.bck_cards))))

    def notify_anti_tribute(self, msg_list, anti_num, anti_pos):
        """
        抗贡阶段，用于通知所有玩家，有人抗贡。其中antiNums的取值与antiPos数组的长度所对应
        :param msg_list: 待发送消息列表
        :param anti_num: 抗贡人数
        :param anti_pos: 抗贡玩家（们）的座位号
        """
        for player in self.players:
            msg_list.append(MsgToBeSend(player.pos, JSONGen.notify_anti_tribute(anti_num, anti_pos)))

    def act(self, msg_list, cur_pos, stage):
        """
        在出牌阶段或进贡阶段或还贡阶段，用于通知对应玩家需要做出动作
        :param msg_list: 待发送消息列表
        :param cur_pos: 需要通知的玩家的座位号
        :param stage: 当前阶段
        局部变量说明：
            player：需要通知的玩家
            opponent：需要通知的玩家的对手
            act_args：函数JSONGen.act所需要的参数
        """
        player = self.players[cur_pos]
        opponent = self.players[(cur_pos + 1) % 4]
        cur_action = self.state.current_action.to_json() if self.state.current_action else None
        greater_action = self.state.greater_action.to_json() if self.state.greater_action else None
        act_args = (
            stage,
            [str(card) for card in player.hand_cards],
            [p.public_info() for p in self.players],
            Card.ALL_RANKS[player.rank],
            Card.ALL_RANKS[opponent.rank],
            self.rank,
            self.state.current_pos,
            cur_action,
            self.state.greater_pos,
            greater_action,
            self.action.action_list
        )
        msg_list.append(MsgToBeSend(cur_pos, JSONGen.act(*act_args)))

    def first_action(self, cur_pos):
        """
        指定座位号的玩家做出先手动作
        :param cur_pos:座位号
        """
        self.action.parse_first_action(self.players[cur_pos].hand_cards, self.players[cur_pos].hearts_num, self.rank)

    def second_action(self, cur_pos):
        """
        指定座位号的玩家做出后手动作
        :param cur_pos: 座位号
        """
        self.action.parse_second_action(
            self.players[cur_pos].hand_cards,
            self.players[cur_pos].hearts_num,
            self.rank,
            self.state.greater_action,
            self.state.rank_order,
            self.state.number_order
        )

    def add_player(self, name, index):
        self.players.append(Player(name, index))

    def start(self, start_pos=None):
        """
        每小局开始前，清空当前动作、当前最大动作，获取最先出牌的玩家（对应代码中的current_pos）的先手动作，并依座位号从小到大的次序将
        （座位号， 手牌信息）的元组添加到待发送的消息列表中，最后添加需要做出先手动作的玩家的消息，并返回待发送的信息
        :param start_pos: 指定某位玩家先出牌
        :return: list[tuple]
        """

        self.reshuffle()
        self.deal()
        # self.record_file = "./{}_{}_{}_{}.data".format(
        #     self.players[0].name, self.players[1].name, time.strftime("%Y%m%d_%H%M%S", time.localtime()), self.times
        # )
        msg_to_be_send = []
        self.over_order.clear()
        self.state.reset()
        cur_pos = start_pos if start_pos else randint(0, 3)
        self.first_action(cur_pos)
        self.notify_beginning(msg_to_be_send)
        self.act(msg_to_be_send, cur_pos, STAGE_PLAY)
        self.state.current_pos = cur_pos
        self.loop = self.play
        return msg_to_be_send

    def play(self, content):
        action = self.action[content["actIndex"]]
        self.state.current_action = action
        self.players[self.state.current_pos].play_area = action
        msg_to_be_send = []
        if action.type == Action.PASS:
            self.notify_play(msg_to_be_send)
            if self.first_action_right_transfer():
                cur_pos = (self.state.greater_pos + 2) % 4
                self.state.clear_action()
                self.first_action(cur_pos)
            else:
                cur_pos = self.next_have_cards_pos()
                if cur_pos == self.state.greater_pos:
                    self.state.clear_action()
                    self.first_action(cur_pos)
                else:
                    self.second_action(cur_pos)
            self.act(msg_to_be_send, cur_pos, STAGE_PLAY)
            self.state.current_pos = cur_pos
            return msg_to_be_send
        else:
            player = self.players[self.state.current_pos]
            player.play_card(action.cards, "H{}".format(self.rank))
            player.play_area = action
            self.state.update_greater_action(action)
            self.notify_play(msg_to_be_send)
            if len(player.hand_cards) == 0:
                self.over_order.add(self.state.current_pos)
                if self.over_order.episode_over():
                    for i in range(4):
                        if len(self.players[i].hand_cards) > 0:
                            self.over_order.add(i)
                    first_pos, _, inc = self.over_order.settlement()
                    self.notify_episode_over(msg_to_be_send)
                    if self.__rank == 14:
                        if (self.rank_belongs + 2) % 4 == first_pos or self.rank_belongs == first_pos:
                            if (first_pos + 2) % 4 == self.over_order.fourth:
                                self.stuck_at_ace(first_pos, (first_pos + 2) % 4, inc, msg_to_be_send)
                            else:
                                self.one_times_over(first_pos, msg_to_be_send)
                        else:
                            self.enter_tribute_stage(first_pos, inc, msg_to_be_send)
                    else:
                        self.enter_tribute_stage(first_pos, inc, msg_to_be_send)
                else:
                    self.next_player_second_action(msg_to_be_send)
            else:
                self.next_player_second_action(msg_to_be_send)
            return msg_to_be_send

    def tribute(self, content=None, msg_to_be_send=None):
        if content:
            msg_to_be_send = []
            tribute_pos, to = self.over_order.find_ship(self.state.current_pos, _ship="tri")
            action = self.action[content["actIndex"]]
            self.over_order.tri_cards.append((tribute_pos, to, action.cards[0]))
            self.players[tribute_pos].play_card(action.cards, "H{}".format(self.rank))
            if self.over_order.index == 0 and len(self.over_order.tri_ship) == 2:
                self.over_order.index += 1
                tribute_pos, _ = self.over_order.tri_ship[-1]
                self.state.clear_action()
                self.action.action_list = self.players[tribute_pos].max_cards(self.rank, "H" + self.rank)
                self.action.valid_range = range(0, len(self.action.action_list))
                self.act(msg_to_be_send, tribute_pos, STAGE_TRIBUTE)
                self.state.current_pos = tribute_pos
                return msg_to_be_send
            else:
                for tribute_tuple in self.over_order.tri_cards:
                    tribute_pos, to, card = tribute_tuple
                    self.players[to].add_card(card, self.rank)
                if len(self.over_order.tri_cards) == 2:
                    # tri_cards: [ [进贡pos, 接受进贡pos, 进贡的牌], [进贡pos, 接受进贡pos, 进贡的牌] ]
                    tribute_a, tribute_b = self.over_order.tri_cards[0], self.over_order.tri_cards[1]
                    tri_pos_a, _, card_str_a = tribute_a
                    tri_pos_b, _, card_str_b = tribute_b
                    cmp_result = cmp2rank(card_str_a[1], card_str_b[1], self.state.rank_order)
                    if cmp_result == 1:
                        self.over_order.first_play = tri_pos_b
                    elif cmp_result == -1:
                        self.over_order.first_play = tri_pos_a
                    else:
                        self.over_order.first_play = self.over_order.fourth
                else:
                    self.over_order.first_play = self.over_order.fourth
                self.notify_tribute(msg_to_be_send)
                self.over_order.index = 0
                back, to = self.over_order.bck_ship[0]
                self.state.current_pos = -1
                self.state.current_action = None
                self.action.action_list = self.players[back].less_than_ten(self.rank)
                self.action.valid_range = range(0, len(self.action.action_list))
                self.act(msg_to_be_send, back, STAGE_BACK)
                self.state.current_pos = back
                self.loop = self.back
                return msg_to_be_send
        else:
            self.notify_beginning(msg_to_be_send)
            fourth_anti = False
            if self.players[self.over_order.fourth].red_joker_num == 2:
                fourth_anti = True
            if len(self.over_order.tri_ship) == 2:
                third_anti = False
                if self.players[self.over_order.third].red_joker_num == 2:
                    third_anti = True
                elif fourth_anti is False and self.players[self.over_order.fourth].red_joker_num == 1 and \
                        self.players[self.over_order.third].red_joker_num == 1:
                    fourth_anti, third_anti = True, True
                if fourth_anti is True or third_anti is True:
                    self.notify_anti_tribute(msg_to_be_send, 2, [self.over_order.third, self.over_order.fourth])
                    self.first_action(self.over_order.first)
                    self.act(msg_to_be_send, self.over_order.first, STAGE_PLAY)
                    self.state.current_pos = self.over_order.first
                    self.loop = self.play
                    self.over_order.clear()
                else:
                    self.action.action_list = self.players[self.over_order.fourth].max_cards(self.rank, "H" + self.rank)
                    self.action.valid_range = range(0, len(self.action.action_list))
                    self.act(msg_to_be_send, self.over_order.fourth, STAGE_TRIBUTE)
                    self.state.current_pos = self.over_order.fourth
                return msg_to_be_send
            else:
                if fourth_anti:
                    self.over_order.tri_ship = []
                    self.over_order.bck_ship = []
                    self.notify_anti_tribute(msg_to_be_send, 1, [self.over_order.fourth])
                    self.first_action(self.over_order.first)
                    self.act(msg_to_be_send, self.over_order.first, STAGE_PLAY)
                    self.state.current_pos = self.over_order.first
                    self.loop = self.play
                    self.over_order.clear()
                else:
                    self.action.action_list = self.players[self.over_order.fourth].max_cards(self.rank, "H" + self.rank)
                    self.action.valid_range = range(0, len(self.action.action_list))
                    self.act(msg_to_be_send, self.over_order.fourth, STAGE_TRIBUTE)
                    self.state.current_pos = self.over_order.fourth
                return msg_to_be_send

    def back(self, content=None):
        back_pos, to = self.over_order.find_ship(self.state.current_pos, _ship="bck")
        action = self.action[content["actIndex"]]
        self.over_order.bck_cards.append((back_pos, to, action.cards[0]))
        self.players[back_pos].play_card(action.cards, "H{}".format(self.rank))
        msg_to_be_send = []
        if self.over_order.index == 0 and len(self.over_order.bck_ship) == 2:
            self.over_order.index += 1
            back_pos, _ = self.over_order.bck_ship[-1]
            self.action.action_list = self.players[back_pos].less_than_ten(back_pos)
            self.action.valid_range = range(0, len(self.action.action_list))
            self.act(msg_to_be_send, back_pos, STAGE_BACK)
            self.state.current_pos = back_pos
        else:
            for bck_tuple in self.over_order.bck_cards:
                _, to, card = bck_tuple
                self.players[to].add_card(card, self.rank)
            self.notify_back(msg_to_be_send)
            self.over_order.index = 0
            self.state.current_pos = -1
            self.first_action(self.over_order.first_play)
            # msg_to_be_send.append(
            #     MsgToBeSend(2, {"type": "notify", "stage": "afterTri", "handCards": self.players[2].hand2json()})
            # )
            self.act(msg_to_be_send, self.over_order.first_play, STAGE_PLAY)
            self.state.current_pos = self.over_order.first_play
            self.loop = self.play
            self.over_order.clear()
        return msg_to_be_send
