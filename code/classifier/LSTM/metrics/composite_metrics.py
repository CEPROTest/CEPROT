# encoding=utf-8
from typing import List

from metrics import BaseCompositeMetric, BaseGenMetric
from metrics.gen_metrics import EMPTY_TOKEN


class UpdateStat(BaseCompositeMetric):
    def eval(self, labels: List[bool], hypos: List[List[List[str]]], references: List[List[str]],
             src_references: List[List[str]], *args, **kwargs):
        assert len(labels) == len(hypos)
        TP_correct = 0
        TP_wrong = 0
        FP_correct = 0
        FP_wrong = 0
        for label, hypo_list, ref, src_ref in zip(labels, hypos, references, src_references):
            hypo = hypo_list[0] if hypo_list[0] else [EMPTY_TOKEN]
            is_equal = BaseGenMetric.is_equal(hypo, ref)
            if label:
                if is_equal:
                    TP_correct += 1
                else:
                    TP_wrong += 1
            else:
                if is_equal:
                    FP_correct += 1
                else:
                    FP_wrong += 1

        return {
            "TP_correct": TP_correct,
            "TP_wrong": TP_wrong,
            "FP_correct": FP_correct,
            "FP_wrong": FP_wrong
        }


class CorrectNum(BaseCompositeMetric):
    def eval(self, labels: List[bool], hypos: List[List[List[str]]], references: List[List[str]],
             src_references: List[List[str]], *args, **kwargs):
        assert len(labels) == len(hypos)
        correct_count = 0
        for label, hypo_list, ref, src_ref in zip(labels, hypos, references, src_references):
            if label:
                hypo = hypo_list[0] if hypo_list[0] else [EMPTY_TOKEN]
                if BaseGenMetric.is_equal(hypo, ref):
                    correct_count += 1
        return correct_count


class WrongNum(BaseCompositeMetric):
    def eval(self, labels: List[bool], hypos: List[List[List[str]]], references: List[List[str]],
             src_references: List[List[str]], *args, **kwargs):
        assert len(labels) == len(hypos)
        wrong_count = 0
        for label, hypo_list, ref, src_ref in zip(labels, hypos, references, src_references):
            if not label:
                hypo = hypo_list[0] if hypo_list[0] else [EMPTY_TOKEN]
                if not BaseGenMetric.is_equal(hypo, ref):
                    wrong_count += 1
        return wrong_count
