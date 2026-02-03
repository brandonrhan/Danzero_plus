# -*- coding: utf-8 -*-
# @Time       : 2020/10/5 14:52
# @Author     : time not yet
# @File       : card.py
# @Description:


class Card(object):

    ALL_SUITS = ('S', 'H', 'C', 'D')
    ALL_RANKS = ('', '', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')

    def __init__(self, suit, rank, digital):
        self.suit = suit
        self.rank = rank
        self.digital = digital

    def __eq__(self, other):
        if isinstance(other, str):
            return self.suit == other[0] and self.rank == other[1]
        else:
            return self.suit == other.suit and self.rank == other.rank

    def __str__(self):
        return self.suit + self.rank

    def __repr__(self):
        return str(self)
