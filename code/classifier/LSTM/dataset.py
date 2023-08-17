# encoding=utf-8
import csv
import difflib
import math
import json
import os
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Iterable, Tuple, Callable, Union
from common import *
import logging

from utils.io import dump_list_to_csv
from vocab import VocabEntry, ExtVocabEntry
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)

def readjson_all(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        d.append(i)
    return d


class AbstractExample(ABC):
    @property
    @abstractmethod
    def src_tokens(self):
        pass

    @property
    @abstractmethod
    def tgt_tokens(self):
        pass


class Example(AbstractExample):
    def __init__(self, instance):
        self._sample_id = instance['sample_id'] if 'sample_id' in instance else None
        self._idx = instance['idx'] if 'idx' in instance else None
        # focal_src focal_tgt
        self._code_change_seqs = instance['code_change_seq'][:300]
        assert self._code_change_seqs is not None
        # 
        self._src_desc_tokens = instance['src_desc_tokens'][:150]
        # self._focal_src_tokens = instancep['focal_src_tokens']
        # NOTE: add START and END marks in tgt_tokens
        # self._tgt_desc_tokens = instance['dst_desc_tokens'][:150]
        # for debugging
        self.src_method = instance['src_method'][:150]
        self.tgt_method = instance['dst_method'][:150]
        self.src_desc = instance['src_desc'][:150]
        self._focal_src_tokens = instance['focal_src_tokens'][:150]
        self._focal_tgt_tokens = instance['focal_tgt_tokens'][:150]

        # self.tgt_desc = instance['dst_desc']
        # for pointer generator
        self.src_ext_vocab = None
        self.code_ext_vocab = None
        self.both_ext_vocab = None
        # for label
        self.label = instance['label'] if 'label' in instance else None

    @classmethod
    def dump_key_mappings(cls):
        return OrderedDict({
            "sample_id": "_sample_id",
            "idx": "_idx",
            "code_change_seq": "_code_change_seqs",
            "src_desc_tokens": "_src_desc_tokens",
            "src_method": "src_method",
            "dst_method": "tgt_method",
            "src_desc": "src_desc",
            "label": "label"
        })

    def dump_to_json(self):
        instance = OrderedDict()
        for key, attr in self.dump_key_mappings().items():
            instance[key] = getattr(self, attr)
        return instance

    def cal_hash_str(self):
        cur_str = ""
        cur_str += str(hash(tuple(tuple(seq) for seq in self._code_change_seqs)))
        cur_str += str(hash(tuple(self._src_desc_tokens)))
        cur_str += str(hash(tuple(self._tgt_desc_tokens)))
        return cur_str

    @staticmethod
    def create_partial_example(instance):
        assert 'code_change_seq' in instance
        assert 'src_desc_tokens' in instance
        instance['sample_id'] = 0
        instance['src_method'] = ""
        instance['dst_method'] = ""
        instance['src_desc'] = ""
        return Example(instance)

    @staticmethod
    def create_zero_example():
        instance = {
            'code_change_seq': [[PADDING, PADDING, UNK]],
            'src_desc_tokens': [PADDING, PADDING],
            'src_method': "",
            'dst_method': "",
            'src_desc': "",
        }
        return Example(instance)

    @property
    def old_code_tokens(self):
        return [seq[0] for seq in self._code_change_seqs]

    @property
    def new_code_tokens(self):
        return [seq[1] for seq in self._code_change_seqs]

    @property
    def edit_actions(self):
        return [seq[2] for seq in self._code_change_seqs]

    @property
    def code_len(self):
        return len(self._code_change_seqs)

    @property
    def src_len(self):
        return len(self.src_tokens)

    @property
    def src_tokens(self):
        """
        used for models
        """
        return self._src_desc_tokens
    # ADD
    @property
    def focal_src_tokens(self):
        """
        used for models
        """
        return self._focal_src_tokens
    
    @property
    def focal_tgt_tokens(self):
        """
        used for models
        """
        return self._focal_tgt_tokens

    @property
    def tgt_in_tokens(self):
        return [TGT_START] + self._tgt_desc_tokens

    @property
    def tgt_out_tokens(self):
        return self._tgt_desc_tokens + [TGT_END]

    @property
    def tgt_tokens(self):
        """
        used for models
        """
        return [TGT_START] + self._tgt_desc_tokens + [TGT_END]

    def get_src_desc_tokens(self) -> List[str]:
        return self._src_desc_tokens

    def get_tgt_desc_tokens(self) -> List[str]:
        return self._tgt_desc_tokens
    
    # ADD
    def get_focal_src_tokens(self) -> List[str]:
        return self._focal_src_tokens
    
    def get_focal_tgt_tokens(self) -> List[str]:
        return self._focal_tgt_tokens

    def get_code_tokens(self):
        code_tokens = []
        for seq in self._code_change_seqs:
            for token in seq[:2]:
                # "" should also be added to the vocab
                code_tokens.append(token)
        return code_tokens

    def get_nl_tokens(self):
        """
        used for build vocab
        """
        return self._src_desc_tokens + self._tgt_desc_tokens

    @property
    def tgt_words_num(self):
        return len(self.tgt_tokens) - 1

    def get_src_ext_vocab(self, base_vocab: VocabEntry = None):
        if not self.src_ext_vocab:
            if not base_vocab:
                raise Exception("Require base_vocab to build src_ext_vocab")
            self.src_ext_vocab = ExtVocabEntry(base_vocab, self.src_tokens)
        return self.src_ext_vocab

    def get_code_ext_vocab(self, base_vocab: VocabEntry = None):
        if not self.code_ext_vocab:
            if not base_vocab:
                raise Exception("Require base_vocab to build code_ext_vocab")
            self.code_ext_vocab = ExtVocabEntry(base_vocab, self.new_code_tokens)
        return self.code_ext_vocab

    def get_both_ext_vocab(self, base_vocab: VocabEntry = None):
        if not self.both_ext_vocab:
            if not base_vocab:
                raise Exception("Require base_vocab to build both_ext_vocab")
            # combine the two tokens
            self.both_ext_vocab = ExtVocabEntry(base_vocab, self.src_tokens + self.new_code_tokens)
        return self.both_ext_vocab

    def dump_keys(self, keys: list):
        new_ex = {}
        for key in keys:
            new_ex[key] = str(getattr(self, key))
            if key in ['src_desc', 'tgt_desc']:
                new_ex[key] = new_ex[key].replace("\n", ' ')
        return new_ex

    def pretty_dump(self):
        diff = list(difflib.unified_diff(self.src_method.splitlines(True), self.tgt_method.splitlines(True)))
        self.diff = "".join(diff)
        new_ex = self.dump_keys(self.pretty_dump_keys())
        read_new_ex = self.dump_keys(self.pretty_dump_read_keys())
        return new_ex, read_new_ex

    @classmethod
    def pretty_dump_keys(cls):
        return ['_sample_id', '_idx', 'diff', 'src_desc', 'tgt_desc', 'label']

    @classmethod
    def pretty_dump_read_keys(cls):
        return ['_sample_id', '_idx', 'diff', 'src_desc']

    @classmethod
    def pretty_dump_examples_to_csv(cls, examples: List, filename: str):
        new_exs = []
        for ex in examples:
            new_ex, _ = ex.pretty_dump()
            new_exs.append(new_ex)
        dump_list_to_csv(new_exs, cls.pretty_dump_keys(), filename)

    @classmethod
    def pretty_dump_examples_with_results_to_csv(cls, examples: List, filename: str, our_name: str, our_results: List[str],
                                                 baselines: List[str], baseline_results: List[List[str]]):
        """
        :param examples: A list of examples
        :param filename: The filename to write
        :param our_name: The name of our approach
        :param our_results: A list of result for dump
        :param baselines: A list of baseline names
        :param baseline_results: A list of list, each sublist contains a result of a baseline. The order of baseline
            results aligns with the order of baseline names.
        :return:
        """
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=cls.pretty_dump_keys() + [our_name] + baselines)
            writer.writeheader()
            for idx, ex in enumerate(examples):
                new_ex, _ = ex.pretty_dump()
                new_ex.update({our_name: our_results[idx]})
                for name, b_r in zip(baselines, baseline_results[idx]):
                    new_ex.update({name: b_r})
                writer.writerow(new_ex)


class Batch(object):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item) -> Example:
        return self.examples[item]

    def __add__(self, other):
        return Batch(self.examples + other.examples)

    @staticmethod
    def create_zero_batch(batch_size: int = 8):
        examples = [Example.create_zero_example() for _ in range(batch_size)]
        return Batch(examples)

    @property
    def tgt_words_num(self) -> int:
        # return sum([e.tgt_words_num for e in self.examples])
        return 0

    @property
    def old_code_tokens(self) -> List[List[str]]:
        return [e.old_code_tokens for e in self.examples]

    @property
    def new_code_tokens(self) -> List[List[str]]:
        return [e.new_code_tokens for e in self.examples]

    @property
    def edit_actions(self) -> List[List[str]]:
        return [e.edit_actions for e in self.examples]

    @property
    def src_tokens(self):
        return [e.src_tokens for e in self.examples]
    #ADD
    @property
    def focal_src_tokens(self):
        return [e.focal_src_tokens for e in self.examples]
    @property
    def focal_tgt_tokens(self):
        return [e.focal_tgt_tokens for e in self.examples]  

    @property
    def tgt_in_tokens(self):
        return [e.tgt_in_tokens for e in self.examples]

    @property
    def tgt_out_tokens(self):
        return [e.tgt_out_tokens for e in self.examples]

    @property
    def tgt_tokens(self):
        return [e.tgt_tokens for e in self.examples]

    @property
    def tgt_action_masks(self):
        return [e.tgt_action_masks for e in self.examples]

    @property
    def tgt_edit_actions(self):
        return [e.tgt_edit_actions for e in self.examples]

    @property
    def tgt_edit_tokens(self):
        return [e.tgt_edit_tokens for e in self.examples]

    def get_code_change_tensors(self, code_vocab: VocabEntry, action_vocab: VocabEntry, device: torch.device):
        code_tensor_a = code_vocab.to_input_tensor(self.old_code_tokens, device)
        code_tensor_b = code_vocab.to_input_tensor(self.new_code_tokens, device)

        edit_tensor = action_vocab.to_input_tensor(self.edit_actions, device)

        return code_tensor_a, code_tensor_b, edit_tensor

    def get_src_tensor(self, vocab: VocabEntry, device: torch.device) -> Tensor:
        return vocab.to_input_tensor(self.src_tokens, device)
    
    # ADD
    def get_focal_src_tensor(self, vocab: VocabEntry, device: torch.device) -> Tensor:
        return vocab.to_input_tensor(self.focal_src_tokens, device)
    
    def get_focal_tgt_tensor(self, vocab: VocabEntry, device: torch.device) -> Tensor:
        return vocab.to_input_tensor(self.focal_tgt_tokens, device)

    def get_tgt_in_tensor(self, vocab: VocabEntry, device: torch.device) -> Tensor:
        return vocab.to_input_tensor(self.tgt_in_tokens, device)

    def get_tgt_out_tensor(self, vocab: VocabEntry, device: torch.device) -> Tensor:
        return vocab.to_input_tensor(self.tgt_out_tokens, device)

    def get_src_ext_tgt_out_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device):
        word_ids = []
        for e in self:
            ext_vocab = e.get_src_ext_vocab(dec_nl_vocab)
            word_ids.append(ext_vocab.words2indices(e.tgt_out_tokens))
        return ids_to_input_tensor(word_ids, dec_nl_vocab[PADDING], device)

    def get_code_ext_tgt_out_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device):
        word_ids = []
        for e in self:
            ext_vocab = e.get_code_ext_vocab(dec_nl_vocab)
            word_ids.append(ext_vocab.words2indices(e.tgt_out_tokens))
        return ids_to_input_tensor(word_ids, dec_nl_vocab[PADDING], device)

    def get_both_ext_tgt_out_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device):
        word_ids = []
        for e in self:
            ext_vocab = e.get_both_ext_vocab(dec_nl_vocab)
            word_ids.append(ext_vocab.words2indices(e.tgt_out_tokens))
        return ids_to_input_tensor(word_ids, dec_nl_vocab[PADDING], device)

    def get_tgt_in_edit_tensors(self, action_vocab: VocabEntry, token_vocab: VocabEntry, device: torch.device) \
            -> Tuple[Tensor, Tensor]:
        tgt_in_ids = [e.get_tgt_ids(action_vocab, token_vocab)[:-1] for e in self.examples]
        tgt_in_tensor = ids_to_input_tensor(tgt_in_ids, token_vocab[PADDING], device)
        tgt_in_action_masks = [e.tgt_action_masks[:-1] for e in self.examples]
        tgt_in_action_mask_tensor = ids_to_input_tensor(tgt_in_action_masks, False, device)
        return tgt_in_tensor, tgt_in_action_mask_tensor

    def get_tgt_out_edit_tensors(self, action_vocab: VocabEntry, token_vocab: VocabEntry, device: torch.device) \
            -> Tuple[Tensor, Tensor]:
        # TODO: here we assume the caller only use code_ext
        tgt_out_ids = [e.get_code_ext_tgt_ids(action_vocab, token_vocab)[1:] for e in self.examples]
        tgt_out_tensor = ids_to_input_tensor(tgt_out_ids, token_vocab[PADDING], device)
        tgt_out_action_masks = [e.tgt_action_masks[1:] for e in self.examples]
        tgt_out_action_mask_tensor = ids_to_input_tensor(tgt_out_action_masks, False, device)
        return tgt_out_tensor, tgt_out_action_mask_tensor

    def get_labels(self):
        return [e.label for e in self.examples]

    def get_label_tensor(self, device):
        return torch.tensor(self.get_labels(), dtype=torch.long, device=device)

    def get_src_lens(self):
        return [len(sent) for sent in self.src_tokens]
    # ADD
    def get_focal_src_lens(self):
        return [len(sent) for sent in self.focal_src_tokens]
    
    def get_focal_tgt_lens(self):
        return [len(sent) for sent in self.focal_tgt_tokens]

    def get_code_lens(self):
        return [e.code_len for e in self.examples]

    def get_src_ext_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device) -> Tensor:
        word_ids = []
        base_vocab = dec_nl_vocab
        for e in self:
            ext_vocab = e.get_src_ext_vocab(base_vocab)
            word_ids.append(ext_vocab.words2indices(e.src_tokens))
        sents_var = ids_to_input_tensor(word_ids, base_vocab[PADDING], device)
        # (src_sent_len, batch_size)
        return sents_var

    def get_code_ext_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device) -> Tensor:
        """
        :param nl_vocab: the vocab of the generated tokens
        :param device:
        :return:
        """
        word_ids = []
        base_vocab = dec_nl_vocab
        for e in self:
            ext_vocab = e.get_code_ext_vocab(base_vocab)
            word_ids.append(ext_vocab.words2indices(e.new_code_tokens))
        sents_var = ids_to_input_tensor(word_ids, base_vocab[PADDING], device)
        # (src_code_len, batch_size)
        return sents_var

    def get_both_ext_tensor(self, dec_nl_vocab: VocabEntry, device: torch.device) -> Tuple[Tensor, Tensor]:
        src_word_ids = []
        code_word_ids = []
        base_vocab = dec_nl_vocab
        for e in self:
            ext_vocab = e.get_both_ext_vocab(base_vocab)
            src_word_ids.append(ext_vocab.words2indices(e.src_tokens))
            code_word_ids.append(ext_vocab.words2indices(e.new_code_tokens))
        src_tensor = ids_to_input_tensor(src_word_ids, base_vocab[PADDING], device)
        code_tensor = ids_to_input_tensor(code_word_ids, base_vocab[PADDING], device)
        return src_tensor, code_tensor

    def get_max_src_ext_size(self) -> int:
        return max([e.get_src_ext_vocab().ext_size for e in self])

    def get_max_code_ext_size(self) -> int:
        return max([e.get_code_ext_vocab().ext_size for e in self])

    def get_max_both_ext_size(self) -> int:
        return max([e.get_both_ext_vocab().ext_size for e in self])

    def construct_encoder_input(self, code_vocab, action_vocab, nl_vocab, device):
        code_tensor_a, code_tensor_b, action_tensor = self.get_code_change_tensors(code_vocab, action_vocab, device)
        src_tensor = self.get_src_tensor(nl_vocab, device)
        # ADD
        focal_src_tensor = self.get_focal_src_tensor(code_vocab, device)
        focal_tgt_tensor = self.get_focal_tgt_tensor(code_vocab, device)

        code_lens = self.get_code_lens()
        src_lens = self.get_src_lens()
        focal_src_lens = self.get_focal_src_lens()
        focal_tgt_lens = self.get_focal_tgt_lens()

        return code_tensor_a, code_tensor_b, action_tensor, code_lens, src_tensor, src_lens, focal_src_tensor, focal_tgt_tensor, focal_src_lens, focal_tgt_lens


class AbstractDataset(ABC):
    @classmethod
    @abstractmethod
    def create_from_file(cls, file_path: str, ExampleClass: Callable):
        pass


class AbstractDataIterator(ABC):
    @abstractmethod
    def train_batch_iter(self, batch_size: int, *args, **kwargs):
        pass

    @abstractmethod
    def infer_batch_iter(self, batch_size: int):
        pass


class Dataset(AbstractDataset, AbstractDataIterator):
    def __init__(self, examples: List[Example]):
        self.examples = examples
# ADD

    @classmethod
    def create_from_file(cls, file_path: str, ExampleClass: Callable = Example):
        examples = []
        # examples_all = readjson_all(file_path)
        with open(file_path, 'r') as f:
            for line in f.readlines():
                examples.append(ExampleClass(json.loads(line)))
        # for line in examples_all:
        #     examples.append(ExampleClass(line))
            
        logging.info("loading {} samples".format(len(examples)))
        return cls(examples)

    def dump_to_jsonl(self, file_path):
        with open(file_path, 'w') as f:
            for ex in self.examples:
                f.write(json.dumps(ex.dump_to_json()))
                f.write('\n')

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)

    def get_code_tokens(self):
        for e in self.examples:
            yield e.get_code_tokens()

    def get_nl_tokens(self):
        for e in self.examples:
            yield e.get_nl_tokens()

    def get_mixed_tokens(self):
        for e in self.examples:
            yield e.get_code_tokens() + e.get_nl_tokens()

    def get_ground_truth(self) -> Iterable[List[str]]:
        for e in self.examples:
            # remove the <s> and </s>
            yield e.get_tgt_desc_tokens()

    def get_src_descs(self) -> Iterable[List[str]]:
        for e in self.examples:
            yield e.get_src_desc_tokens()

    def _batch_iter(self, batch_size: int, shuffle: bool, sort_by_length: bool) -> Batch:
        batch_num = math.ceil(len(self) / batch_size)
        # logging.info(f"\n\n\nkkkkdlldldlldldl\n\n")
        # logging.info(f"{batch_num}")
        index_array = list(range(len(self)))
        # logging.info(f"{index_array}")
        if shuffle:
            np.random.shuffle(index_array)
        logging.info("_batch_iter")
        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            examples = [self[idx] for idx in indices]
            # logging.info(f"{examples}")
            if sort_by_length:
                examples = sorted(examples, key=lambda e: len(e.src_tokens), reverse=True)
            yield Batch(examples)

    def sample_examples(self, n: int) -> List[Example]:
        return random.sample(self.examples, n)

    def train_batch_iter(self, batch_size: int, shuffle: bool, *args, **kwargs) -> Iterable[Batch]:
        # for batch in self._batch_iter(batch_size, shuffle=shuffle, sort_by_length=True):
        for batch in self._batch_iter(batch_size, shuffle=shuffle, sort_by_length=False):
            t = type(batch)
            # logging.info(f"{t}")
            # print(batch.examples)
            yield batch

    def infer_batch_iter(self, batch_size):
        for batch in self._batch_iter(batch_size, shuffle=False, sort_by_length=False):
            yield batch

    def equal(self, another: "Dataset"):
        if len(self) != len(another):
            return False

        hash_set1 = []
        for ex in self:
            hash_set1.append(ex.cal_hash_str())
        hash_set1.sort()

        hash_set2 = []
        for ex in another:
            hash_set2.append(ex.cal_hash_str())
        hash_set2.sort()

        if hash_set1 != hash_set2:
            return False

        return True


class LargeDataset(AbstractDataset, AbstractDataIterator):
    def __init__(self, file_path: str, ExampleClass: Callable):
        self.len = None
        self.file_path = file_path
        self.ExampleClass = ExampleClass
        self.old_cursor = None
        self.file_cursor = None
        self.renew_file_cursor()

    def renew_file_cursor(self):
        self.old_cursor = self.file_cursor
        self.file_cursor = open(self.file_path, 'r')

    def restore_file_cursor(self):
        # close current file cursor
        if self.file_cursor:
            self.file_cursor.close()
        # restore old cursor or create a new cursor
        if self.old_cursor:
            self.file_cursor = self.old_cursor
            self.old_cursor = None
        else:
            self.file_cursor = open(self.file_path, 'r')

    def close(self):
        if self.file_cursor:
            self.file_cursor.close()
        if self.old_cursor:
            self.old_cursor.close()

    @classmethod
    def create_from_file(cls, file_path: str, ExampleClass: Callable = Example):
        return cls(file_path, ExampleClass)

    def __len__(self):
        if self.len is None:
            self.renew_file_cursor()
            self.len = 0
            while True:
                line = self.file_cursor.readline().strip()
                if not line:
                    break
                self.len += 1
            self.restore_file_cursor()
        return self.len

    def __iter__(self):
        while True:
            line = self.file_cursor.readline().strip()
            if not line:
                break
            yield self.ExampleClass(json.loads(line))

    def _batch_iter(self, batch_size: int, sort_by_length: bool) -> Iterable[Batch]:
        self.renew_file_cursor()
        example_count = 0
        batch_examples = []
        for example in self:
            batch_examples.append(example)
            example_count += 1
            if example_count == batch_size:
                if sort_by_length:
                    batch_examples.sort(key=lambda e: len(e.src_tokens), reverse=True)
                yield Batch(batch_examples)
                example_count = 0
                batch_examples = []
        if example_count > 0:
            if sort_by_length:
                batch_examples.sort(key=lambda e: len(e.src_tokens), reverse=True)
            yield Batch(batch_examples)
        self.restore_file_cursor()

    def train_batch_iter(self, batch_size: int, *args, **kwargs) -> Iterable[Batch]:
        for batch in self._batch_iter(batch_size, sort_by_length=True):
            yield batch

    def infer_batch_iter(self, batch_size: int):
        for batch in self._batch_iter(batch_size, sort_by_length=False):
            yield batch

    def get_code_tokens(self):
        self.renew_file_cursor()
        for e in self:
            yield e.get_code_tokens()
        self.restore_file_cursor()

    def get_nl_tokens(self):
        self.renew_file_cursor()
        for e in self:
            yield e.get_nl_tokens()
        self.restore_file_cursor()

    def get_mixed_tokens(self):
        self.renew_file_cursor()
        for e in self:
            yield e.get_code_tokens() + e.get_nl_tokens()
        self.restore_file_cursor()

    def get_ground_truth(self) -> Iterable[List[str]]:
        self.renew_file_cursor()
        for e in self:
            # remove the <s> and </s>
            yield e.get_tgt_desc_tokens()
        self.restore_file_cursor()

    def get_src_descs(self) -> Iterable[List[str]]:
        self.renew_file_cursor()
        for e in self:
            yield e.get_src_desc_tokens()
        self.restore_file_cursor()


class BalancedSampleDataset(AbstractDataIterator):
    def __init__(self, neg_dataset: LargeDataset, pos_dataset: Dataset):
        self.neg_dataset = neg_dataset
        self.pos_dataset = pos_dataset

    def train_batch_iter(self, batch_size: int, shuffle: bool, *args, **kwargs):
        pos_batch_size = batch_size // 2
        neg_batch_size = batch_size - pos_batch_size
        for neg_batch in self.neg_dataset.train_batch_iter(neg_batch_size, shuffle=shuffle):
            pos_examples = self.pos_dataset.sample_examples(pos_batch_size)
            examples = neg_batch.examples + pos_examples
            # sort by length
            examples = sorted(examples, key=lambda e: len(e.src_tokens), reverse=True)
            yield Batch(examples)

    def infer_batch_iter(self, batch_size: int):
        raise NotImplementedError

    def close(self):
        self.neg_dataset.close()


def get_dataset_class(file_path: str) -> Callable:
    size_g = os.path.getsize(file_path) / 1024 ** 3
    if size_g > 2:
        return LargeDataset
    else:
        return Dataset


def load_dataset_from_file(file_path: str, ExampleClass: Callable = Example) -> Union[Dataset, LargeDataset]:
    DatasetClass = get_dataset_class(file_path)
    return DatasetClass.create_from_file(file_path, ExampleClass)