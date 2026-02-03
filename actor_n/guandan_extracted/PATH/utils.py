# -*- coding: utf-8 -*-
# @Time       : 2020/10/5 22:04
# @Author     : time not yet
# @File       : utils.py
# @Description: 提供辅助功能

RANK = ('', '', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')

CARD_TABLE = {
    "PASS": 1,
    "S2": 2,
    "S3": 3,
    "S4": 4,
    "S5": 5,
    "S6": 6,
    "S7": 7,
    "S8": 8,
    "S9": 9,
    "ST": 10,
    "SJ": 11,
    "SQ": 12,
    "SK": 13,
    "SA": 14,

    "H2": 15,
    "H3": 16,
    "H4": 17,
    "H5": 18,
    "H6": 19,
    "H7": 20,
    "H8": 21,
    "H9": 22,
    "HT": 23,
    "HJ": 24,
    "HQ": 25,
    "HK": 26,
    "HA": 27,

    "C2": 28,
    "C3": 29,
    "C4": 30,
    "C5": 31,
    "C6": 32,
    "C7": 33,
    "C8": 34,
    "C9": 35,
    "CT": 36,
    "CJ": 37,
    "CQ": 38,
    "CK": 39,
    "CA": 40,

    "D2": 41,
    "D3": 42,
    "D4": 43,
    "D5": 44,
    "D6": 45,
    "D7": 46,
    "D8": 47,
    "D9": 48,
    "DT": 49,
    "DJ": 50,
    "DQ": 51,
    "DK": 52,
    "DA": 53,

    "SB": 54,
    "HR": 55
}

RANK_TABLE = {
    "H2": 2,
    "H3": 3,
    "H4": 4,
    "H5": 5,
    "H6": 6,
    "H7": 7,
    "H8": 8,
    "H9": 9,
    "HT": 10,
    "HJ": 11,
    "HQ": 12,
    "HK": 13,
    "HA": 14
}

COOMON_RANK_TABLE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
    "B": 15,
    "R": 16
}

CARD_DIGITAL_TABLE = {
    "SA": 0x010e,
    "S2": 0x0102,
    "S3": 0x0103,
    "S4": 0x0104,
    "S5": 0x0105,
    "S6": 0x0106,
    "S7": 0x0107,
    "S8": 0x0108,
    "S9": 0x0109,
    "ST": 0x010a,
    "SJ": 0x010b,
    "SQ": 0x010c,
    "SK": 0x010d,

    "HA": 0x020e,
    "H2": 0x0202,
    "H3": 0x0203,
    "H4": 0x0204,
    "H5": 0x0205,
    "H6": 0x0206,
    "H7": 0x0207,
    "H8": 0x0208,
    "H9": 0x0209,
    "HT": 0x020a,
    "HJ": 0x020b,
    "HQ": 0x020c,
    "HK": 0x020d,

    "CA": 0x030e,
    "C2": 0x0302,
    "C3": 0x0303,
    "C4": 0x0304,
    "C5": 0x0305,
    "C6": 0x0306,
    "C7": 0x0307,
    "C8": 0x0308,
    "C9": 0x0309,
    "CT": 0x030a,
    "CJ": 0x030b,
    "CQ": 0x030c,
    "CK": 0x030d,

    "DA": 0x040e,
    "D2": 0x0402,
    "D3": 0x0403,
    "D4": 0x0404,
    "D5": 0x0405,
    "D6": 0x0406,
    "D7": 0x0407,
    "D8": 0x0408,
    "D9": 0x0409,
    "DT": 0x040a,
    "DJ": 0x040b,
    "DQ": 0x040c,
    "DK": 0x040d,

    "SB": 0x0110,
    "HR": 0x0211
}


def cmp2cards(card, other, current_rank, flag):
    """
    比较两张卡牌，如果前者小于后者，返回True，反之False
    :param card: 卡牌A
    :param other: 卡牌B
    :param current_rank: 当前等级
    :param flag: 是否忽略花色
    :return: True or False
    """
    card_point = 15 if card.rank == current_rank else card.digital & 0x00FF
    other_point = 15 if other.rank == current_rank else other.digital & 0x00FF
    if card_point < other_point:
        return True
    elif card_point == other_point:
        if flag:
            return False
        else:
            return card.digital & 0xFF00 < other.digital & 0xFF00
    else:
        return False


def cmp2rank(rank, other, order):
    """
    比较两个rank的大小，如果前者大于后者返回-1，后者大于前者返回1，相等返回0
    :param rank: rank A
    :param other: rank B
    :param order: rank的点数大小
    :return: int
    """
    if order[rank] > order[other]:
        return -1
    elif order[rank] < order[other]:
        return 1
    else:
        return 0


def separate_cards(hand_cards, repeat_flag):
    """
    将一个有序手牌中的不同点数的单张分别划分到一个列表
    :param hand_cards: 玩家座位号
    :param repeat_flag: 列表中是否包含相同点数、相同花色的单张
    :return: List[List[Card]]
    """
    foo = hand_cards[0]
    result = []
    cards = []
    for card in hand_cards:

        # 因为手牌是有序的，那么利用哨兵法，只要和哨兵比较的结果不同说明不属同一列表
        if card.rank != foo.rank:
            result.append(cards)
            foo = card
            cards = [card, ]
        else:
            if repeat_flag:
                foo = card
                cards.append(card)
            elif foo == card and len(cards) != 0:
                continue
            else:
                foo = card
                cards.append(card)
    result.append(cards)
    return result


def draw_same_rank_cards(same_cards, cards, n, name):
    """
    使用游标选择指定的数量的相同点数的卡牌（即用于选择两对、三张、炸弹等牌型）
    TEST  如左所示，采用游标法（也可视做二进制法）来生成指定元素个数的组合，1表示所指向的元素
    1101
    :param same_cards: 相同点数的卡牌的列表
    :param cards: 包含相同点数的卡牌列表
    :param n: 指定数量
    :param name: 牌型名称
    :return: List[list[Card]]
    """
    inx_ptr = [0 for _ in range(n)]                               # 下标数组
    for i in range(n - 1):                                        # 最后一位下标从尾部开始
        inx_ptr[i] = i
    inx_ptr[n - 1] = len(cards) - 1                               # 尾标作为移动的基础，并指向最后一个元素的下标
    flag = True
    while inx_ptr[0] <= len(cards) - n and flag:                  # 跳出循环条件之一：当主标之后的元素不够选时
        foo = []
        for inx in inx_ptr:
            foo.append(str(cards[inx]))
        same_cards.append([name, foo[0][1], foo])

        if cards[inx_ptr[n - 1] - 1] == cards[inx_ptr[n - 1]]:    # 加入元素后，尾标后移
            inx_ptr[n - 1] -= 2                                   # 跳过相同元素应该后移2
        else:
            inx_ptr[n - 1] -= 1                                   # 不跳过相同元素后移1

        if inx_ptr[n - 1] <= inx_ptr[n - 2]:                      # 如果尾标大于前标，说明前标需要进行移动
            inx_ptr[n - 1] = len(cards) - 1                       # 先重置尾标
            i = n - 2                                             # 从后往前依次判断是哪个标需要移动
            while i >= 0:
                inx_ptr[i] += 1                                   # 先后移
                if inx_ptr[i] == inx_ptr[i + 1]:                  # 若后移后碰到了后标，说明该标的前标需要后移，以此类推
                    inx_ptr[i] -= 1                               # 所以重置该标的位置
                    if i == 0:                                    # 如果是头标后移碰到后标，说明大循环结束
                        flag = False
                    i -= 1
                    continue
                elif cards[inx_ptr[i]] == cards[inx_ptr[i] - 1]:  # 未碰到后标，若下一元素与移动前相同，则跳过，该标自增
                    inx_ptr[i] += 1
                    if inx_ptr[i] == inx_ptr[i + 1]:              # 若再次碰到尾标，同样，说明该标前标需要后移１
                        inx_ptr[i] -= 1                           # 同样，头标碰到后标，大循环结束
                        if i == 0:
                            flag = False
                        i -= 1
                        continue
                    else:                                         # 跳过相同元素后即为正确的位置，重置该标后的所有标
                        for j in range(i + 1, n - 1):
                            inx_ptr[j] = inx_ptr[j - 1] + 1
                        break
                else:                                             # 后移1后未碰到尾标或后面元素与移动前相同，后移2没有碰到尾标
                    for j in range(i + 1, n - 1):                 # 重置该标后的所有标
                        inx_ptr[j] = inx_ptr[j - 1] + 1
                    break


class OverOrder(object):
    """
    管理完牌次序, 并根据完牌次序, 给出进贡和还贡的关系
    """

    def __init__(self):
        self.order = []  # 记录完牌次序
        self.tri_ship = []  # 进贡关系
        self.bck_ship = []  # 还贡关系
        self.tri_cards = []  # 记录谁进贡了什么牌
        self.bck_cards = []  # 记录谁还贡了什么牌
        self.index = 0  # 记录以及还贡或者进贡了几个人
        self.first_play = 0

    def add(self, pos):
        """
        将完牌的玩家的座位号添加到列表中
        :param pos: 座位号
        :return: None
        """
        self.order.append(pos)

    def settlement(self):
        """
        根据完牌次序进行结算，并保存进贡还贡关系，返回结算后对应玩家需要升几级
        :return: tuple(int, int, int)
        """
        if (self.order[0] + 2) % 4 == self.order[1]:
            # 双下，顺时针进贡，逆时针还贡
            inc = 3
            order = [0, 1, 2, 3]
            self.tri_ship.append((self.order[-1], order[self.order[-1] - 1]))
            self.tri_ship.append((self.order[-2], order[self.order[-2] - 1]))
            self.bck_ship.append((order[self.order[-1] - 1], self.order[-1]))
            self.bck_ship.append((order[self.order[-2] - 1], self.order[-2]))
        elif (self.order[0] + 2) % 4 == self.order[2]:
            inc = 2
            self.tri_ship.append((self.order[-1], self.order[0]))
            self.bck_ship.append((self.order[0], self.order[-1]))
        else:
            inc = 1
            self.tri_ship.append((self.order[-1], self.order[0]))
            self.bck_ship.append((self.order[0], self.order[-1]))
        return self.order[0], (self.order[0] + 2) % 4, inc

    def episode_over(self):
        """
        判断座位号为pos的玩家的队友是否已经在完牌序列中
        :return:
        """
        if (self.order[-1] + 2) % 4 in self.order:
            return True
        return False

    def clear(self):
        self.first_play = -1
        self.index = 0
        self.order.clear()
        self.tri_ship.clear()
        self.bck_ship.clear()
        self.tri_cards.clear()
        self.bck_cards.clear()

    def find_ship(self, pos, _ship):
        ship = getattr(self, "{}_ship".format(_ship))
        for s in ship:
            if pos in s:
                return s

    @property
    def first(self):
        return self.order[0]

    @property
    def second(self):
        return self.order[1]

    @property
    def third(self):
        return self.order[2]

    @property
    def fourth(self):
        return self.order[3]


class JSONGen(object):

    @staticmethod
    def notify_beginning(hand_cards, index, cur_rank, pos_2_rank, oppo_rank):
        return {
            "type": "notify",
            "stage": "beginning",
            "handCards": [str(card) for card in hand_cards],
            "myPos": index,
            "curRank": cur_rank,
            "selfRank": pos_2_rank,
            "oppoRank": oppo_rank
        }

    @staticmethod
    def notify_play(cur_pos, cur_action, greater_pos, greater_action):
        return {
            "type": "notify",
            "stage": "play",
            "curPos": cur_pos,
            "curAction": cur_action,
            "greaterPos": greater_pos,
            "greaterAction": greater_action
        }

    @staticmethod
    def notify_episode_over(cur_rank, over_order, rest_cards):
        return {
            "type": "notify",
            "stage": "episodeOver",
            "curRank": cur_rank,
            "order": over_order,
            "restCards": rest_cards
        }

    @staticmethod
    def notify_game_over(cur_times, setting_times):
        return {
            "type": "notify",
            "stage": "gameOver",
            "curTimes": cur_times,
            "settingTimes": setting_times
        }

    @staticmethod
    def notify_game_result(respective_wins, respective_draws):
        return {
            "type": "notify",
            "stage": "gameResult",
            "victoryNum": respective_wins,
            "draws": respective_draws
        }

    @staticmethod
    def notify_tribute(tribute_cards):
        return {
            "type": "notify",
            "stage": "tribute",
            "result": tribute_cards
        }

    @staticmethod
    def notify_back(back_cards):
        return {
            "type": "notify",
            "stage": "back",
            "result": back_cards
        }

    @staticmethod
    def notify_anti_tribute(anti_num, anti_pos):
        return {
            "type": "notify",
            "stage": "anti-tribute",
            "antiNum": anti_num,
            "antiPos": anti_pos
        }

    @staticmethod
    def act(stage, hand_cards, public_info, self, oppo, cur, cur_pos, c_act, g_pos, g_act, action_list):
        return {
            "type": "act",
            "stage": stage,
            "handCards": hand_cards,
            "publicInfo": public_info,
            "selfRank": self,
            "oppoRank": oppo,
            "curRank": cur,
            "curPos": cur_pos,
            "curAction": c_act,
            "greaterPos": g_pos,
            "greaterAction": g_act,
            "actionList": action_list,
            "indexRange": len(action_list) - 1
        }
