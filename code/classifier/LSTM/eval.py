# encoding=utf-8
import os
import json
from argparse import ArgumentParser, Namespace

from common import *
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple, Union, Callable

from config import load_yaml_config
from dataset import load_dataset_from_file
from utils.io import load_json_file
from utils.tokenizer import Tokenizer
from infer import CompositeInfer
from metrics.clf_metrics import ClfMetric
from metrics.composite_metrics import CorrectNum, WrongNum, UpdateStat
from metrics.gen_metrics import Accuracy, Recall, NLGMetrics, GLEU, EditDistance, UpdateNum

logging.basicConfig(level=logging.INFO)


class BaseEvaluator(ABC):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def evaluate(self):
        pass


class BaseGenEvaluator(BaseEvaluator, ABC):
    METRICS = "accuracy,recall,distance,nlg,gleu,update_num"

    def __init__(self, metrics: str = METRICS, **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics

    @abstractmethod
    def load_hypos_and_refs(self, test_set_file: str, result_file: str, normalize_hypo: bool, *args, **kwargs) \
            -> Tuple[List[List[List[str]]], List[List[str]], List[List[str]]]:
        pass


class Evaluator(BaseGenEvaluator):
    METRIC_MAP = {
        "accuracy": Accuracy(),
        "recall": Recall(k=5),
        "distance": EditDistance(),
        "nlg": NLGMetrics(),
        "gleu": GLEU(),
        "update_num": UpdateNum()
    }

    def __init__(self,
                 result_file: str,
                 test_set: str,
                 metric_map: dict = None,
                 no_lemma: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.result_file = result_file
        self.test_set = test_set
        self.metric_map = metric_map if metric_map else self.METRIC_MAP
        self.no_lemma = no_lemma
        self.nlp = StanfordNLPTool() if not no_lemma else None

    def load_hypos(self, result_file: str) -> List[List[List[str]]]:
        results = load_json_file(result_file)
        return self.load_hypos_raw(results)

    @staticmethod
    def load_hypos_raw(results) -> List[List[List[str]]]:
        assert results[0] == [] or (type(results[0][0][0]) == list and type(results[0][0][1] == float)), \
            "Each example should have a list of Hypothesis. Please prepare your result like " \
            "[Hypothesis(desc, score), ...]"
        hypos = [[hypo[0] for hypo in r] for r in results]
        return hypos

    @staticmethod
    def normalize_hypos(hypos, src_references):
        new_hypos = []
        for hypo_list, src_sent in zip(hypos, src_references):
            if not hypo_list:
                logging.error("find empty hypo list")
                hypo_list = [src_sent]
            new_hypos.append(hypo_list)
        return new_hypos

    def load_hypos_and_refs(self, test_set_file: str, result_file: str, normalize_hypo: bool = True, *args, **kwargs) \
            -> Tuple[List[List[List[str]]], List[List[str]], List[List[str]]]:
        test_set = load_dataset_from_file(test_set_file)
        references = list(test_set.get_ground_truth())
        src_references = list(test_set.get_src_descs())
        hypos = self.load_hypos(result_file)
        if normalize_hypo:
            hypos = self.normalize_hypos(hypos, src_references)

        return hypos, references, src_references

    def _load_lemmas(self, origin: List, file_path: str, try_load: bool) -> List:
        if try_load and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        nlp = self.nlp
        if isinstance(origin[0][0], str):
            lemmas = [nlp.lemmatize_list(sent) for sent in origin]
        elif isinstance(origin[0][0], Iterable):
            lemmas = [[nlp.lemmatize_list(s) if s else [""] for s in sents] for sents in origin]
        else:
            raise TypeError("origin[0][0] should be str or Iterable, but is {}".format(type(origin[0])))
        with open(file_path, 'w') as f:
            json.dump(lemmas, f)
        return lemmas

    def prepare(self, hypos: List[List[List[str]]], references: List[List[str]], src_references: List[List[str]]) \
            -> Tuple[List[List[List[str]]], List[List[str]], List[List[str]]]:
        file_path = self.result_file + '.l.hypos'
        lemma_hypos = self._load_lemmas(hypos, file_path, False)
        file_path = self.test_set + '.l.refs'
        lemma_refs = self._load_lemmas(references, file_path, False)
        file_path = self.test_set + '.l.src_refs'
        lemma_src_refs = self._load_lemmas(src_references, file_path, False)

        return lemma_hypos, lemma_refs, lemma_src_refs

    def cal_metrics(self, metrics: Iterable[str], hypos: List[List[List[str]]], references: List[List[str]],
                    src_references: List[List[str]]):
        results = {}
        for metric in metrics:
            instance = self.metric_map[metric.lower()]
            results[metric] = instance.eval(hypos, references, src_references)
        return results

    def evaluate(self):
        metrics = self.metrics.split(',')
        hypos, references, src_references = self.load_hypos_and_refs(self.test_set, self.result_file)
        assert type(hypos[0][0]) == type(references[0])
        results = self.cal_metrics(metrics, hypos, references, src_references)
        logging.info(results)
        return results

    def cal_scores(self):
        metrics = self.metrics.split(',')
        hypos, references, src_references = self.load_hypos_and_refs(self.test_set, self.result_file)
        assert type(hypos[0][0]) == type(references[0])
        scores = {}
        for metric in metrics:
            instance = self.metric_map[metric.lower()]
            if metric != "distance":
                scores[metric] = instance.cal_scores(hypos, references, src_references)
            else:
                src_distances, hypo_distances, rel_distances = instance.cal_scores(hypos, references, src_references)
                scores['ED'] = hypo_distances
                scores['RED'] = rel_distances
        return scores

    def test(self):
        hypos, references, src_references = self.load_hypos_and_refs(self.test_set, self.result_file)
        lemma_hypos, lemma_refs, lemma_src_refs = self.prepare(hypos, references, src_references)
        for i, (h, r, sr, lh, lr, lsr) in enumerate(zip(hypos, references, src_references, lemma_hypos, lemma_refs,
                                                        lemma_src_refs)):
            h = h[0]
            lh = lh[0]
            if h == r and lh != lr:
                print("not equal: {}".format(i))

            if EditDistance.relative_distance(h, r, sr) < EditDistance.relative_distance(lh, lr, lsr):
                print("lemma is worse:")
                print("h: {}".format(h))
                print("r: {}".format(r))
                print("sr: {}".format(sr))
                print("lh: {}".format(lh))
                print("lr: {}".format(lr))
                print("lsr: {}".format(lsr))


def recover_desc_tokens(tokens: List[str], action_masks: List[int], src_desc: List[str]):
    cur_index = 0
    desc_tokens = []
    error_action = False
    for token, mask in zip(tokens, action_masks):
        if mask == 0:
            desc_tokens.append(token)
            continue
        if token == ACTION_2_TGT_ACTION['equal']:
            if cur_index < len(src_desc):
                desc_tokens.append(src_desc[cur_index])
                cur_index += 1
            else:
                error_action = True
            continue
        if token in (ACTION_2_TGT_ACTION['delete'], ACTION_2_TGT_ACTION['replace']):
            cur_index += 1
            continue
    return desc_tokens, error_action


class FracoEvaluator(Evaluator):
    def __init__(self,
                 **kwargs):
        super(FracoEvaluator, self).__init__(**kwargs)
        self.matched_count = 0

    @staticmethod
    def prepare_fraco_result_sent(r: dict) -> List[str]:
        return Tokenizer.tokenize_desc_with_con(r['result'])

    def load_hypos(self, result_file) -> List[List[List[str]]]:
        with open(result_file, 'r') as f:
            results = json.load(f)
        hypos = []
        for r in results:
            if r['matched']:
                self.matched_count += 1
            sent = self.prepare_fraco_result_sent(r)
            hypos.append([sent])
        return hypos

    def evaluate_with_raw_desc(self):
        match_count = 0
        correct_count = 0

        with open(self.result_file, 'r') as f:
            results = json.load(f)
        with open(self.test_set, 'r') as f:
            lines = f.readlines()
        assert len(results) == len(lines)
        for line, r in zip(lines, results):
            example = json.loads(line)
            if r['matched']:
                match_count += 1
                if r['result'].strip() == example['dst_desc'].strip():
                    correct_count += 1
            else:
                assert r['result'] == example['src_desc']
        return correct_count, match_count


class ClfEvaluator(BaseEvaluator):
    THRESHOLD = 0.5

    def __init__(self,
                 result_file: str,
                 test_set: str,
                 threshold: float = THRESHOLD,
                 **kwargs):
        super().__init__(**kwargs)
        self.result_file = result_file
        self.test_set = test_set
        self.threshold = threshold
        self.probs = None
        self.labels = None

    def load_probs(self) -> np.array:
        if self.probs is None:
            self.probs = torch.load(self.result_file).numpy()
        return self.probs

    def load_labels(self) -> np.array:
        if self.labels is None:
            test_set = load_dataset_from_file(self.test_set)
            labels = []
            for e in test_set:
                labels.append(int(e.label))
            self.labels = np.array(labels)
        return self.labels

    @classmethod
    def cal_metrics(cls, probs, labels, threshold):
        metric = ClfMetric(threshold)
        result = metric.eval(probs, labels)
        return result

    def evaluate(self):
        probs, labels = self.load_probs(), self.load_labels()
        assert len(probs) == len(labels)
        threshold = float(self.threshold)
        result = self.cal_metrics(probs, labels, threshold)
        logging.info(result)
        return result


class FracoClfEvaluator(ClfEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_probs(self):
        if self.probs is None:
            with open(self.result_file, 'r') as f:
                results = json.load(f)
            probs = []
            for r in results:
                if r['matched']:
                    probs.append([0., 1.])
                else:
                    probs.append([1., 0.])
            self.probs = np.array(probs)
        return self.probs


class CompositeEvaluator(BaseGenEvaluator):
    CLF_EVAL_CLASS = ClfEvaluator
    GEN_EVAL_CLASS = Evaluator
    COM_METRIC_MAP = {
        "update_stat": UpdateStat(),
        "correct_num": CorrectNum(),
        "wrong_num": WrongNum()
    }

    @classmethod
    def add_trainer_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--clf-eval-class', type=get_evaluator_by_name, default=cls.CLF_EVAL_CLASS,
                            help="Class used for evaluating classifier")
        parser.add_argument('--gen-eval-class', type=get_evaluator_by_name, default=cls.GEN_EVAL_CLASS,
                            help="Class used for evaluating updater")
        return parser

    def __init__(self,
                 result_file: str,
                 test_set: str,
                 gen_metrics: str = Evaluator.METRICS,
                 no_lemma: bool = True,
                 clf_eval_class: Union[str, Callable] = CLF_EVAL_CLASS,
                 gen_eval_class: Union[str, Callable] = GEN_EVAL_CLASS,
                 **kwargs):
        super().__init__(**kwargs)
        self.result_file = result_file
        self.test_set = test_set
        logging.info("Clf Evaluator: {}".format(clf_eval_class))
        if isinstance(clf_eval_class, str):
            clf_eval_class = get_evaluator_by_name(clf_eval_class)
        if isinstance(gen_eval_class, str):
            gen_eval_class = get_evaluator_by_name(gen_eval_class)

        if clf_eval_class != FracoClfEvaluator:
            result_file = CompositeInfer.get_probs_file(self.result_file)
            self.logger.info("Classifier result file: {}".format(result_file))
        else:
            result_file = self.result_file
        self.clf_args = Namespace(
            result_file=result_file,
            test_set=test_set,
            **kwargs
        )
        self.clf_evaluator = clf_eval_class(**vars(self.clf_args))

        logging.info("Gen Evaluator: {}".format(gen_eval_class))
        if gen_eval_class != FracoEvaluator:
            result_file = CompositeInfer.get_hypos_file(self.result_file)
            self.logger.info("Generator result file: {}".format(result_file))
        else:
            result_file = self.result_file
        self.gen_args = Namespace(
            result_file=result_file,
            test_set=test_set,
            metrics=gen_metrics,
            no_lemma=no_lemma,
            **kwargs
        )
        self.gen_evaluator = gen_eval_class(**vars(self.gen_args))

        self.com_metric_map = self.COM_METRIC_MAP

    def load_hypos_and_refs(self, test_set_file: str, result_file: str, normalize_hypo: bool, *args, **kwargs) -> \
            Tuple[List[List[List[str]]], List[List[str]], List[List[str]]]:
        return self.gen_evaluator.load_hypos_and_refs(test_set_file, result_file, normalize_hypo=normalize_hypo,
                                                      *args, **kwargs)

    def evaluate(self):
        logging.info("Evaluating Classifier")
        logging.warning("For now, we do not support threshold")
        clf_result = self.clf_evaluator.evaluate()
        probs = self.clf_evaluator.load_probs()
        labels = self.clf_evaluator.load_labels()

        metrics = self.metrics.split(',')
        hypos, references, src_references = \
            self.load_hypos_and_refs(self.gen_args.test_set, self.gen_args.result_file,
                                     normalize_hypo=False)
        pred_labels = probs.argmax(axis=-1)
        assert len(pred_labels) == len(hypos)
        tp_hypos, tp_refs, tp_src_refs = [], [], []
        gen_hypos, gen_refs, gen_src_refs = [], [], []

        # the labels for the samples predicted to be positive
        gen_labels = []
        for pred, label, hypo, ref, src in zip(pred_labels, labels, hypos, references, src_references):
            if pred == 1:
                # predicted to be positive
                assert hypo != []
                gen_labels.append(label)
                gen_hypos.append(hypo)
                gen_refs.append(ref)
                gen_src_refs.append(src)
                if label == 1:
                    # TP
                    tp_hypos.append(hypo)
                    tp_refs.append(ref)
                    tp_src_refs.append(src)
        logging.info("Evaluating Updater: TP")
        logging.info("TP = {}".format(len(tp_hypos)))
        tp_result = self.gen_evaluator.cal_metrics(metrics, tp_hypos, tp_refs, tp_src_refs)
        logging.info(tp_result)

        logging.info("Evaluating Updater: TP + FP")
        logging.info("TP + FP = {}".format(len(gen_hypos)))
        gen_result = self.gen_evaluator.cal_metrics(metrics, gen_hypos, gen_refs, gen_src_refs)
        logging.info(gen_result)

        logging.info("Evaluating total precision")
        gen_correct_num = tp_result['accuracy']['correct_count']
        total_update_num = labels.sum()
        total_accuracy = gen_correct_num * 1.0 / total_update_num
        logging.info(total_accuracy)

        logging.info("Evaluating composite metric")
        com_result = {}
        for key, metric in self.com_metric_map.items():
            com_result[key] = metric.eval(gen_labels,
                                          gen_hypos,
                                          gen_refs,
                                          gen_src_refs)
        logging.info(com_result)
        result = {
            'clf_result': clf_result,
            'gen_result': gen_result,
            'TP': len(tp_hypos),
            'TP_FP': len(gen_hypos),
            'com_result': com_result,
        }

        return result


def get_evaluator_by_name(name: str):
    return globals()[name]


def build_evaluator(log_dir: str,
                    eval_class: str,
                    no_lemma: bool = True,
                    log: bool = True,
                    **kwargs) -> BaseEvaluator:
    log_file = os.path.join(log_dir, "eval.log")
    if log:
        setup_logger(logging.root, log_file, logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    EvalClass = get_evaluator_by_name(eval_class)
    kwargs['result_file'] = os.path.join(log_dir, kwargs['result_file'])
    evaluator = EvalClass(no_lemma=no_lemma, **kwargs)
    return evaluator


def build_evaluator_from_args(configs: dict,
                              log_dir: str,
                              no_lemma: bool = True,
                              log: bool = True,
                              **kwargs):
    configs = configs.pop('eval')
    configs.update(kwargs)

    evaluator = build_evaluator(log_dir, no_lemma=no_lemma, log=log, **configs)
    return evaluator


def build_evaluator_from_config(config_file: str,
                                log_dir: str,
                                no_lemma: bool = True,
                                log: bool = True,
                                **kwargs):
    all_configs = load_yaml_config(config_file)
    return build_evaluator_from_args(all_configs, log_dir, no_lemma, log, **kwargs)


def eval_from_args(configs: dict,
                   log_dir: str,
                   no_lemma: bool = True,
                   log: bool = True,
                   **kwargs):
    evaluator = build_evaluator_from_args(configs, log_dir, no_lemma, log, **kwargs)
    return evaluator.evaluate()


def eval_from_config(config_file: str,
                     log_dir: str,
                     no_lemma: bool = True,
                     log: bool = True,
                     **kwargs):
    evaluator = build_evaluator_from_config(config_file, log_dir, no_lemma, log, **kwargs)
    return evaluator.evaluate()


def load_updater_result(name: str, config_file: str, log_dir: str) -> List[List[str]]:
    configs = load_yaml_config(config_file)
    result_file = os.path.join(log_dir, configs['eval']['result_file'])

    if not name.lower().startswith("fraco"):
        with open(result_file, 'r') as f:
            rs = json.load(f)
        return [r[0][0] for r in rs]

    with open(result_file, 'r') as f:
        fraoc_results = json.load(f)
    return [FracoEvaluator.prepare_fraco_result_sent(r) for r in fraoc_results]


def load_detector_result(name: str, config_file: str, log_dir: str):
    configs = load_yaml_config(config_file)
    result_file = os.path.join(log_dir, configs['eval']['result_file'])

    if not name.lower().startswith("fraco"):
        return torch.load(result_file).numpy().argmax(axis=-1)
    # for fraco
    result = []
    with open(result_file, 'r') as f:
        fraoc_results = json.load(f)
        for r in fraoc_results:
            pred = 1 if r['matched'] else 0
            result.append(pred)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True,
                        help="dir for storing log and model")
    parser.add_argument('--config', type=str, required=True,
                        help="config file")
    args = parser.parse_args()

    eval_from_config(config_file=args.config, log_dir=args.log_dir)


if __name__ == '__main__':
    main()
