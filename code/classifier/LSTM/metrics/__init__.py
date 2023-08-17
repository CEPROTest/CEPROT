from abc import ABC, abstractmethod
from typing import Iterable, List
import numpy as np

from common import recover_desc


class BaseClfMetric(ABC):
    """
    Base class for classification metrics
    """

    @abstractmethod
    def eval(self, probs: np.array, labels: np.array):
        pass


class BaseGenMetric(ABC):
    """
    Base class for generation metrics
    """

    @abstractmethod
    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> float:
        """
        :param hypos: each hypo contains k sents, for accuracy, only use the first sent, for recall, use k sents
        :param references: the dst desc sents
        :param src_references: the src desc sents
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def cal_scores(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
                   src_references: Iterable[List[str]]):
        pass

    @staticmethod
    def is_equal(hypo: List[str], ref: List[str]):
        if hypo == ref:
            return True
        if ref[-1] in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_'.split() and ref[:-1] == hypo:
            return True
        return False

    @staticmethod
    def prepare_split_sent(tokens: List[str]) -> List[str]:
        return recover_desc(tokens).split(" ")


class BaseCompositeMetric(ABC):
    @abstractmethod
    def eval(self, labels: Iterable[bool], hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs):
        """
        :param labels: the labels of each sample that is predicted to be positive
        :param hypos: generated comments
        :param references: reference comments
        :param src_references: old comments
        :param args:
        :param kwargs:
        :return:
        """
        pass
