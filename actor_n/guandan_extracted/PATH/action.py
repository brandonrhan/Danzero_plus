# -*- coding: utf-8 -*-
# @Time       : 2020/10/5 15:04
# @Author     : time not yet
# @File       : action.py
# @Description:

from collections import Counter, defaultdict
from itertools import product

from utils import CARD_DIGITAL_TABLE, separate_cards, draw_same_rank_cards
from card import Card


def extract_single(hand_cards, _rank=None, order=None):
    """
    获取有序手牌中的单张（不包含重复单张），同样利用哨兵法。与separate_cards不同的是，extract_single将所有单张放在一
    个列表中，而不是分点数存放
    :param hand_cards: 某玩家的手牌
    :param _rank: 是否筛选掉小于rank的单张
    :param order: 一个从rank映射到整型的字典（单张大小根据当前所打的等级进行比较）
    :return: list[["Single", "rank", ['SW']], ...]
    """
    foo = hand_cards[0]
    if _rank:
        if order[foo.rank] > order[_rank]:
            single = [[Action.SINGLE, foo.rank, [str(foo)]], ]
        else:
            single = list()
    else:
        single = [[Action.SINGLE, foo.rank, [str(foo)]], ]
    index = 0
    while index != len(hand_cards):
        if foo == hand_cards[index]:
            index += 1
            continue
        else:
            foo = hand_cards[index]
            if _rank:
                if order[foo.rank] > order[_rank]:
                    single.append([Action.SINGLE, foo.rank, [str(foo)]])
            else:
                single.append([Action.SINGLE, foo.rank, [str(foo)]])
    return single


def extract_same_cards(hand_cards, hearts_num, current_rank, same_cnt, _rank=None, order=None):
    """
    调用separate_cards获取某位玩家的手牌的不同点数卡牌的列表，再调用draw_same_rank_cards来获取某位玩家指定数量的相同点数的牌。（对子、炸弹）
    :param hand_cards: 手牌
    :param hearts_num: 红桃配的数量
    :param current_rank: 当前等级
    :param same_cnt: 相同点数的卡牌的数量
    :param _rank: 是否筛选掉小于rank的组合
    :param order: 一个从rank映射到整型的字典（对子、炸弹的大小根据当前所打的等级进行比较）
    :return: list
    """

    if same_cnt == 2:
        name = Action.PAIR
    elif same_cnt == 3:
        name = Action.TRIPS
    else:
        name = Action.BOMB
    cards = separate_cards(hand_cards, repeat_flag=True)
    same_cards = []
    uni_cnt = 0                                    # 向列表中添加万用牌的数量
    for card_list in cards:
        if _rank and order[card_list[0].rank] <= order[_rank]:
            continue
        if hearts_num == 1:       # 计算uni_cnt取值
            uni_cnt = 1
        elif hearts_num > 1:
            uni_cnt = 1 if same_cnt == 2 else 2

        # 改动1
        if card_list[-1].rank not in ('B', 'R', current_rank):
            for _ in range(uni_cnt):
                card_list.append(Card('H', current_rank, CARD_DIGITAL_TABLE['H' + current_rank]))

        if len(card_list) >= same_cnt:
            draw_same_rank_cards(same_cards, card_list, same_cnt, name=name)

    if same_cnt == 4 and name == Action.BOMB:
        if hand_cards[-4:] == ['SB', 'SB', 'HR', 'HR']:
            same_cards.append([Action.BOMB, "JOKER", ['SB', 'SB', 'HR', 'HR']])
    return same_cards


def extract_straight(hand_cards, hearts_num, current_rank, flush=False, _rank=None, order=None):
    """
    调用separate_cards获取某位玩家手中不同点数的卡牌集合，再用游标法提取该玩家手中的顺子
    :param hand_cards: 玩家手牌
    :param hearts_num: 红桃配的数量
    :param current_rank: 当前等级
    :param flush: 所提取的是否为同花顺
    :param _rank: 是否筛选掉小于rank的顺子
    :param order: 一个从rank映射到整型的字典（同花顺的大小根据牌原始的rank进行比较）
    :return: list[str]
    """

    if len(hand_cards) < 5:
        return []

    result = []

    # 分别对应从A到大小王，如下标ranks_inx[1]就表示A, 10-T, 11-J

    ranks_inx = [-1 for _ in range(18)]
    cards_set = separate_cards(hand_cards, repeat_flag=False)

    # ranks_inx记录对应点数在cards_set中的下标
    for i in range(len(cards_set)):
        foo = cards_set[i][0]

        # 顺子可以从A开始，对A进行单独处理
        if foo.digital & 0x00FF == 14:
            ranks_inx[1] = i
        ranks_inx[cards_set[i][0].digital & 0x00FF] = i

    # 对万用牌进行处理，有万用牌则添加进各个点数列表中
    if hearts_num:
        for _cards in cards_set:
            if _cards[0].rank == current_rank or _cards[0].rank == 'B' or _cards[0].rank == 'R':
                continue
            _cards.append(Card('H', current_rank, CARD_DIGITAL_TABLE['H' + current_rank]))

        # 单独补上一个万用牌的列表，并让所空缺的点数指向该下标
        cards_set.append([Card('H', current_rank, CARD_DIGITAL_TABLE['H' + current_rank]), ])
    else:
        # 没有万用牌则需要补一个哨兵的列表，并让空缺的点数指向该下标
        cards_set.append([Card('H', 'R', CARD_DIGITAL_TABLE['H' + current_rank])])

    # 指向万用牌和指向哨兵是互斥的，所以不是if就是else
    for i in range(18):
        if ranks_inx[i] == -1:
            ranks_inx[i] = len(cards_set) - 1

    key = 1

    # rank游标
    r_ptr = [0 for _ in range(5)]

    # index游标
    inx_ptr = [0 for _ in range(5)]
    for i in range(5):
        r_ptr[i] = ranks_inx[key + i]

    # 顺子最大到10
    while key < 11:
        if _rank and order[cards_set[r_ptr[0]][inx_ptr[0]].rank] <= order[_rank]:
            key += 1
            for k in range(5):
                r_ptr[k] = ranks_inx[key + k]
                inx_ptr[k] = 0
            continue

        # 哨兵牌数量
        _guard_cnt = 0

        # 万用牌数量
        _uni_cnt = 0
        foo = []
        for i in range(5):
            temp = cards_set[r_ptr[i]][inx_ptr[i]]
            if temp.rank == 'R':
                _guard_cnt += 1
            elif temp.rank == current_rank and temp.suit == 'H':
                _uni_cnt += 1
            foo.append(str(temp))

        # 只要有1张哨兵牌说明顺子断层，不符合顺子要求
        if _guard_cnt > 0:
            key += 1
            for i in range(5):
                inx_ptr[i] = 0
                r_ptr[i] = ranks_inx[key + i]
            continue
        elif _uni_cnt > hearts_num:
            pass
        else:

            # 顺子的rank为rank标数组的第一个元素所指代的rank
            first_card = cards_set[r_ptr[0]][0]
            hearts_card = "H" + current_rank
            rank_mapping = ('', 'A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K')
            # 如果第一张牌是红桃配
            if first_card == hearts_card:
                second_card = cards_set[r_ptr[1]][0]
                if second_card == hearts_card:
                    rank = rank_mapping[rank_mapping.index(cards_set[r_ptr[2]][0].rank) - 2]
                else:
                    rank = rank_mapping[rank_mapping.index(second_card.rank) - 1]
            else:
                rank = first_card.rank

            temp = set()
            for card_str in foo:
                temp.add(card_str[0])
            flush_flag = False
            if len(temp) == 1:
                flush_flag = True
            elif len(temp) == 2 and hearts_card in foo:
                __foo = [card_str for card_str in foo if card_str[0] == 'H']
                if __foo == [hearts_card, ] or __foo == [hearts_card, hearts_card, ]:
                    flush_flag = True

            if flush:
                if flush_flag:
                    result.append([Action.STRAIGHT_FLUSH, rank, foo])
            else:
                if not flush_flag:
                    result.append([Action.STRAIGHT, rank, foo])

        # 移动游标，for倒序输出，每次步长-1
        for i in range(4, -1, -1):

            # 如果其中一个标后移1碰到边界, 需要移动它的前标，若是头标碰界，则需要移动rank标
            if inx_ptr[i] + 1 > len(cards_set[r_ptr[i]]) - 1:
                inx_ptr[i] = 0

                # 说明需要移动rank标
                if i == 0:
                    key += 1
                    for k in range(5):
                        r_ptr[k] = ranks_inx[key + k]
                    break
            else:
                inx_ptr[i] += 1
                break
    # 筛选红心配导致
    result_new = []
    if _rank:
        for r in result:
            if r[1] > _rank:
                result_new.append(r)
    else:
        result_new = result
    return result_new


def extract_three_two(hand_cards, hearts_num, current_rank, _rank=None, order=None):
    """
    获取某位玩家手中的三带二牌型
    extract_straight负责获取顺子和同花顺
    extract_same_cards通过组合的方式获取三连对、钢板、三带二的牌型
    :param hand_cards: 手牌
    :param hearts_num: 红桃配的数量
    :param current_rank: 当前等级
    :param _rank: 是否筛选掉小于rank的三带二
    :param order: 一个从rank映射到整型的字典（三带二的大小根据牌当前所打的等级进行比较）
    :return: list
    """

    three = extract_same_cards(hand_cards, hearts_num, current_rank, 3, _rank, order)
    two = extract_same_cards(hand_cards, hearts_num, current_rank, 2)
    three_two = []
    for three_cards in three:
        _, three_rank, cards = three_cards
        if _rank and order[three_rank] <= order[_rank]:
            continue
        for two_cards in two:
            _, two_rank, pair = two_cards
            if three_rank == two_rank:
                continue
            temp_three_two = cards + pair
            counter = Counter(temp_three_two)
            hearts_card = "H{}".format(current_rank)
            if hearts_card in counter and counter[hearts_card] > hearts_num:
                continue
            else:
                three_two.append([Action.THREE_WITH_TWO, three_rank, temp_three_two])
    return three_two


def extract_three_pair(hand_cards, hearts_num, current_rank, _rank=None, order=None):
    """
    三连对牌型提取
    :param hand_cards: 手牌
    :param hearts_num: 红桃配的数量
    :param current_rank: 当前等级
    :param order: 一个从rank映射到整型的字典（三带二的大小根据牌当前所打的等级进行比较）
    :param _rank: 是否筛选掉小于rank的三连对
    :return: dict[str: list]
    """
    # pair = extract_same_cards(hand_cards, hearts_num, current_rank, 2, _rank, order)
    pair = extract_same_cards(hand_cards, hearts_num, current_rank, 2)
    temp_dict = defaultdict(list)
    for p in pair:
        _, rank, cards = p
        temp_dict[rank].append(cards)
    three_pair = []
    ranks = ('A23', '234', '345', '456', '567', '678', '789', '89T', '9TJ', 'TJQ', 'JQK', 'QKA')
    for rank in ranks:
        if _rank and order[rank[0]] <= order[_rank]:
            continue
        if rank[0] in temp_dict and rank[1] in temp_dict and rank[2] in temp_dict:
            for pair_a, pair_b, pair_c in product(temp_dict[rank[0]], temp_dict[rank[1]], temp_dict[rank[2]]):
                hearts_card = 'H' + current_rank
                result = pair_a + pair_b + pair_c
                temp_counter = Counter(result)
                if hearts_card in temp_counter and temp_counter[hearts_card] > hearts_num:
                    continue
                else:
                    three_pair.append([Action.THREE_PAIR, rank[0], result])
    return three_pair


def extract_two_trips(hand_cards, hearts_num, current_rank, _rank=None, order=None):
    """
    钢板牌型提取
    :param hand_cards: 手牌
    :param hearts_num: 红桃配的数量
    :param current_rank: 当前等级
    :param order: 一个从rank映射到整型的字典（三带二的大小根据牌当前所打的等级进行比较）
    :param _rank: 是否筛选掉小于rank的钢板
    :return: dict[str: list]
    """

    trips = extract_same_cards(hand_cards, hearts_num, current_rank, 3, _rank, order)
    temp_dict = defaultdict(list)
    for t in trips:
        _, rank, cards = t
        temp_dict[rank].append(cards)

    two_trips = []
    ranks = ('A2', '23', '34', '45', '56', '67', '78', '89', '9T', 'TJ', 'JQ', 'QK', 'KA')
    for rank in ranks:
        if _rank and order[rank[0]] <= order[_rank]:
            continue
        if rank[0] in temp_dict and rank[1] in temp_dict:
            for trip_a, trip_b in product(temp_dict[rank[0]], temp_dict[rank[1]]):
                hearts_card = 'H' + current_rank
                result = trip_a + trip_b
                temp_counter = Counter(result)
                if hearts_card in temp_counter and temp_counter[hearts_card] > hearts_num:
                    continue
                else:
                    two_trips.append([Action.TWO_TRIPS, rank[0], result])
    return two_trips


class Action(object):

    SINGLE = "Single"
    PAIR = "Pair"
    TRIPS = "Trips"
    THREE_PAIR = "ThreePair"
    THREE_WITH_TWO = "ThreeWithTwo"
    TWO_TRIPS = "TwoTrips"
    STRAIGHT = "Straight"
    STRAIGHT_FLUSH = "StraightFlush"
    BOMB = "Bomb"
    PASS = "PASS"

    def __init__(self, _type=None, _rank=None, _cards=None):
        self._type = _type
        self._rank = _rank
        self._cards = _cards

    def update(self, _type, _rank, _cards):
        self._type = _type
        self._rank = _rank
        self._cards = _cards

    def to_json(self):
        return [self._type, self._rank, self._cards]

    def clear(self):
        self._type = None
        self._rank = None
        self._cards = None

    @property
    def type(self):
        return self._type

    @property
    def rank(self):
        return self._rank

    @property
    def cards(self):
        return self._cards


class ActionList(object):

    def __init__(self):
        self.valid_range = range(0, 1)
        self.action_list = []

    def __len__(self):
        return len(self.action_list)

    def __getitem__(self, item):
        return Action(*self.action_list[item])

    def parse_first_action(self, hand_cards, hearts_num, current_rank):
        """
        根据传递的手牌，获取其中的先手动作，并更新action_list和valid_range
        :param hand_cards: 手牌
        :param hearts_num: 红桃配的数量
        :param current_rank: 当前等级
        :return: list[type, rank, [cards]]
        """
        if len(hand_cards) == 0:
            self.action_list = []
            return
        # hand_cards, hearts_num, current_rank, same_cnt, _rank = None, order = None
        self.action_list.clear()
        self.action_list.extend(extract_single(hand_cards))
        self.action_list.extend(extract_same_cards(hand_cards, hearts_num, current_rank, 2))
        self.action_list.extend(extract_same_cards(hand_cards, hearts_num, current_rank, 3))
        self.action_list.extend(extract_three_pair(hand_cards, hearts_num, current_rank))
        self.action_list.extend(extract_three_two(hand_cards, hearts_num, current_rank))
        self.action_list.extend(extract_two_trips(hand_cards, hearts_num, current_rank))
        self.action_list.extend(extract_straight(hand_cards, hearts_num, current_rank))
        self.action_list.extend(extract_straight(hand_cards, hearts_num, current_rank, flush=True))
        for i in range(4, 11):
            self.action_list.extend(extract_same_cards(hand_cards, hearts_num, current_rank, i))
        self.valid_range = range(0, len(self.action_list))

    def parse_second_action(self, hand_cards, hearts_num, current_rank, greater_action, rank_order, number_order):
        """
        根据传递的手牌，获取其中的后手动作，并更新action_list和valid_range
        :param hand_cards: 手牌
        :param hearts_num: 红桃配的数量
        :param current_rank: 当前等级
        :param greater_action: 最大动作
        :param rank_order: 一个从rank映射到整型的字典（单张，对子，炸弹，三带二的大小根据牌当前所打的等级进行比较）
        :param number_order: 一个从rank映射到整型的字典（连对，钢板，顺子的大小根据牌当前所打的等级进行比较）
        :return:
        """
        self.action_list.clear()
        self.action_list.append([Action.PASS, Action.PASS, Action.PASS])
        add_boom, add_flush = True, True
        rank = greater_action.rank
        if greater_action.type == Action.SINGLE:
            self.action_list.extend(extract_single(hand_cards, _rank=rank, order=rank_order))

        elif greater_action.type == Action.PAIR:
            self.action_list.extend(extract_same_cards(hand_cards, hearts_num, current_rank, 2, rank, rank_order))

        elif greater_action.type == Action.TRIPS:
            self.action_list.extend(extract_same_cards(hand_cards, hearts_num, current_rank, 3, rank, rank_order))

        elif greater_action.type == Action.THREE_PAIR:
            self.action_list.extend(extract_three_pair(hand_cards, hearts_num, current_rank, rank, number_order))

        elif greater_action.type == Action.THREE_WITH_TWO:
            self.action_list.extend(extract_three_two(hand_cards, hearts_num, current_rank, rank, rank_order))

        elif greater_action.type == Action.TWO_TRIPS:
            self.action_list.extend(extract_two_trips(hand_cards, hearts_num, current_rank, rank, number_order))

        elif greater_action.type == Action.STRAIGHT:
            if rank == 'T':
                pass
            else:
                self.action_list.extend(
                    extract_straight(hand_cards, hearts_num, current_rank, False, rank, number_order)
                )

        elif greater_action.type == Action.STRAIGHT_FLUSH:
            if rank == 'T':
                pass
            else:
                self.action_list.extend(
                    extract_straight(hand_cards, hearts_num, current_rank, True, rank, number_order)
                )
            for i in range(6, 11):
                self.action_list.extend(extract_same_cards(hand_cards, hearts_num, current_rank, i))
            add_boom, add_flush = False, False

        elif greater_action.type == Action.BOMB:
            add_boom, add_flush = False, False
            if rank == "JOKER":
                pass
            else:
                bomb_len = len(greater_action.cards)
                self.action_list.extend(
                    extract_same_cards(hand_cards, hearts_num, current_rank, bomb_len, rank, rank_order)
                )
                for i in range(bomb_len + 1, 11):
                    self.action_list.extend(
                        extract_same_cards(hand_cards, hearts_num, current_rank, i)
                    )
                # 同花顺or癞子同花顺，可以压5张及以下的炸弹(修改-yg)
                if bomb_len <= 5:
                    self.action_list.extend(
                        extract_straight(hand_cards, hearts_num, current_rank, True)
                    )

        if add_boom:
            for i in range(4, 11):
                self.action_list.extend(extract_same_cards(hand_cards, hearts_num, current_rank, i))

        if add_flush:
            self.action_list.extend(extract_straight(hand_cards, hearts_num, current_rank, flush=True))

        if hand_cards[-4:] == ['SB', 'SB', 'HR', 'HR']:
            self.action_list.append([Action.BOMB, "JOKER", ['SB', 'SB', 'HR', 'HR']])

        self.valid_range = range(0, len(self.action_list))
