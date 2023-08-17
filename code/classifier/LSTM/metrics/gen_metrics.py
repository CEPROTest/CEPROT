import logging
import os
import subprocess
import time
import numpy as np
from collections import OrderedDict
from typing import Iterable, List, Tuple

from nlgeval import NLGEval

from common import recover_desc
from metrics import BaseGenMetric
from utils.edit import word_level_edit_distance


EMPTY_TOKEN = '<empty>'
GLEU_CMD = "./gleu/scripts/compute_gleu"


class Accuracy(BaseGenMetric):
    def __init__(self, *args, **kwargs):
        super(Accuracy, self).__init__(*args, **kwargs)
        self.correct_count = 0

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]] = None, *args, **kwargs) -> dict:
        correct = 0
        total = 0
        for hypo_list, ref in zip(hypos, references):
            hypo = hypo_list[0] if hypo_list[0] else [EMPTY_TOKEN]
            total += 1
            assert (type(hypo[0]) == str)
            assert (type(ref[0]) == str)
            if self.is_equal(hypo, ref):
                correct += 1
        return {'accuracy': correct / total, 'correct_count': correct, 'total_count': total}

    def cal_scores(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
                   src_references: Iterable[List[str]]) -> List:
        scores = []
        for hypo_list, ref in zip(hypos, references):
            hypo = hypo_list[0] if hypo_list[0] else [EMPTY_TOKEN]
            if self.is_equal(hypo, ref):
                scores.append(1)
            else:
                scores.append(0)
        assert len(scores) == len(hypos)
        return scores


class Recall(BaseGenMetric):
    def __init__(self, k: int = 5, *args, **kwargs):
        super(Recall, self).__init__(*args, **kwargs)
        self.k = k

    def has_correct(self, hypo_list, ref):
        for hypo in hypo_list[:self.k]:
            if self.is_equal(hypo, ref):
                return True
        return False

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]] = None, *args, **kwargs) -> float:
        total = 0
        correct = 0
        for hypo_list, ref in zip(hypos, references):
            total += 1
            if self.has_correct(hypo_list, ref):
                correct += 1
        return correct / total

    def cal_scores(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
                   src_references: Iterable[List[str]]) -> List[int]:
        scores = []
        for hypo_list, ref in zip(hypos, references):
            if self.has_correct(hypo_list, ref):
                scores.append(1)
            else:
                scores.append(0)
        return scores


class NLGMetrics(BaseGenMetric):
    def __init__(self, *args, **kwargs):
        super(NLGMetrics, self).__init__(*args, **kwargs)
        self.nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)

    @staticmethod
    def prepare_sent(tokens: List[str]) -> str:
        return recover_desc(tokens)

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> OrderedDict:
        # List[str]
        first_hypos = [self.prepare_sent(hypo_list[0]) for hypo_list in hypos]
        src_ref_strs = [self.prepare_sent(src_ref) for src_ref in src_references]
        # List[List[str]]
        references_lists = [[self.prepare_sent(ref) for ref in references]]
        # distinct
        metrics_dict = self.nlgeval.compute_metrics(references_lists, first_hypos)
        # relative improve
        src_metrics_dict = self.nlgeval.compute_metrics(references_lists, src_ref_strs)
        relative_metrics_dict = OrderedDict({})
        for key in metrics_dict:
            relative_metrics_dict[key] = (metrics_dict[key] - src_metrics_dict[key]) / src_metrics_dict[key]
        return OrderedDict({
            'hypo': metrics_dict,
            'src': src_metrics_dict,
            'relative': relative_metrics_dict
        })

    def cal_scores(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
                   src_references: Iterable[List[str]]):
        return None


class GLEU(BaseGenMetric):
    def __init__(self, *args, **kwargs):
        super(GLEU, self).__init__(*args, **kwargs)

    @staticmethod
    def prepare_sent(tokens: List[str]) -> str:
        return recover_desc(tokens)

    @staticmethod
    def compute_gleu(hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
                     src_references: Iterable[List[str]]) -> List[str]:
        hypo_sents = []
        ref_sents = []
        src_sents = []
        for hypo_list, ref, src_ref in zip(hypos, references, src_references):
            hypo_sents.append(GLEU.prepare_sent(hypo_list[0]))
            ref_sents.append(GLEU.prepare_sent(ref))
            src_sents.append(GLEU.prepare_sent(src_ref))
        sent_lists = [hypo_sents, ref_sents, src_sents]
        prefixes = ["hypo", "ref", "src"]
        time_str = str(time.time())
        suffix = ".txt"
        pathes = []

        def _file_path(prefix, time_str, suffix):
            return prefix + time_str + suffix

        for sent_list, prefix in zip(sent_lists, prefixes):
            path = _file_path(prefix, time_str, suffix)
            with open(path, 'w') as f:
                f.write("\n".join(sent_list))
                f.write("\n")
            pathes.append(path)
        # run compute_gleu
        cmd = "python2 " + GLEU_CMD + " -s {} -r {} -o {} -n 4 -d"
        cmd = cmd.format(pathes[2], pathes[1], pathes[0])
        output = subprocess.check_output(cmd.split()).decode("utf-8")
        lines = [l.strip() for l in output.split('\n') if l.strip()]

        for path in pathes:
            os.remove(path)

        return lines

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> float:
        lines = self.compute_gleu(hypos, references, src_references)
        logging.info("GLEU output: {}".format(lines[-1]))
        score = float(lines[-1].split()[0])
        return score

    def cal_scores(self, hypos: List[List[List[str]]], references: List[List[str]],
                   src_references: List[List[str]]) -> List[float]:
        lines = self.compute_gleu(hypos, references, src_references)
        start_idx = 0
        while start_idx < len(lines):
            if lines[start_idx][0] == '0':
                break
            start_idx += 1

        scores = []
        for idx in range(len(hypos)):
            terms = lines[start_idx + idx].split()
            scores.append(float(terms[1]))
        return scores


class EditDistance(BaseGenMetric):
    def __init__(self, *args, **kwargs):
        super(EditDistance, self).__init__(*args, **kwargs)

    @staticmethod
    def edit_distance(sent1: List[str], sent2: List[str]) -> int:
        return word_level_edit_distance(sent1, sent2)

    @classmethod
    def relative_distance(cls, src_ref_dis, hypo_ref_dis):
        if src_ref_dis == 0:
            if hypo_ref_dis == 0:
                return 0
            else:
                return hypo_ref_dis
        return hypo_ref_dis / src_ref_dis

    @classmethod
    def cal_distances(cls, hypo, ref, src_ref):
        hypo_ref_dis = cls.edit_distance(hypo, ref)
        src_ref_dis = cls.edit_distance(src_ref, ref)
        rel_dis = cls.relative_distance(src_ref_dis, hypo_ref_dis)
        return hypo_ref_dis, src_ref_dis, rel_dis

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> dict:
        src_distances, hypo_distances, rel_distances = self.cal_scores(hypos, references, src_references)
        rel_dis = float(np.mean(rel_distances))
        src_dis = float(np.mean(src_distances))
        hypo_dis = float(np.mean(hypo_distances))
        return {"rel_distance": rel_dis, "src_distance": src_dis, "hypo_distance": hypo_dis}

    def cal_scores(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
                   src_references: Iterable[List[str]]) -> Tuple[List, List, List]:
        src_distances = []
        hypo_distances = []
        rel_distances = []
        for idx, (hypo_list, ref, src_ref) in enumerate(zip(hypos, references, src_references)):
            hypo = hypo_list[0]
            hypo_ref_dis, src_ref_dis, rel_dis = self.cal_distances(hypo, ref, src_ref)
            src_distances.append(src_ref_dis)
            hypo_distances.append(hypo_ref_dis)
            rel_distances.append(rel_dis)
        return src_distances, hypo_distances, rel_distances


class UpdateNum(BaseGenMetric):
    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> float:
        count = 0
        for hypo_list, ref, src_ref in zip(hypos, references, src_references):
            hypo = hypo_list[0] if hypo_list[0] else [EMPTY_TOKEN]
            assert (type(hypo[0]) == str)
            assert (type(ref[0]) == str)
            if hypo != src_ref:
                count += 1
        return count

    def cal_scores(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
                   src_references: Iterable[List[str]]):
        raise NotImplementedError("cal_scores is not implemented in UpdateNum")
