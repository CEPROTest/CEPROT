# encoding=utf-8
import os

import fire
import vocab


def build_vocab_raw(data_dir: str, ft_model: str, em_file: str, vocab_class: str, vocab_file: str, vocab_size: int):
    arg_str = "--train-set {train_set} " \
              "--use-ft " \
              "--ft-model {ft_model} " \
              "--embedding-file {em_file} " \
              "--vocab-class {vocab_class} " \
              "--size {vocab_size} " \
              "{vocab_file}"
    # build vocab
    arg_list = arg_str.format(
        train_set=os.path.join(data_dir, "train.jsonl"),
        ft_model=ft_model,
        em_file=os.path.join(data_dir, em_file),
        vocab_class=vocab_class,
        vocab_size=vocab_size,
        vocab_file=os.path.join(data_dir, vocab_file)
    ).split()
    vocab.main(arg_list)


def build_cup_vocab(data_dir="data/cup2_updater_dataset",
                    ft_model="~/fastText/cc.en.300.bin",
                    max_vocab_size=50000):
    build_vocab_raw(data_dir, ft_model, "vocab_embeddings.pkl", "Vocab", "vocab.json", max_vocab_size)
    build_vocab_raw(data_dir, ft_model, "mix_vocab_embeddings.pkl", "MixVocab", "mix_vocab.json", max_vocab_size)


def build_ocd_vocab(data_dir="data/cup2_dataset",
                    ft_model="~/fastText/cc.en.300.bin",
                    vocab_prefix="",
                    max_vocab_size=100000):
    build_vocab_raw(data_dir, ft_model, vocab_prefix + "vocab_embeddings.pkl",
                    "Vocab", vocab_prefix + "vocab.json", max_vocab_size)
    build_vocab_raw(data_dir, ft_model, vocab_prefix + "mix_vocab_embeddings.pkl",
                    "MixVocab", vocab_prefix + "mix_vocab.json", max_vocab_size)


if __name__ == '__main__':
    fire.Fire({
        "build_cup_vocab": build_cup_vocab,
        "build_ocd_vocab": build_ocd_vocab
    })
