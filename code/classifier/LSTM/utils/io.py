import csv
import json
import os
from typing import List, Iterable, Dict


def dump_set_to_jsonl(dataset: List, filename, indent: int = None):
    with open(filename, 'w') as f:
        for example in dataset:
            f.write(json.dumps(example, indent=indent))
            f.write('\n')


def load_jsonl_set(filename: str) -> List[Dict]:
    dataset = []
    with open(filename, 'r') as in_f:
        for line in in_f.readlines():
            dataset.append(json.loads(line))
    return dataset


def iter_jsonl_set(filename: str) -> Iterable[Dict]:
    with open(filename, 'r') as in_f:
        while True:
            line = in_f.readline()
            if not line:
                break
            yield json.loads(line)


def load_json_file(filename: str) -> Dict:
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def dump_sentences(sentences: List[str], filename: str):
    with open(filename, 'w') as f:
        for sent in sentences:
            f.write(sent)
            f.write("\n")
        f.flush()


def dump_list_to_csv(instances: List[Dict], fieldnames: List[str], filename: str):
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ins in instances:
            writer.writerow(ins)


def get_detector_metric_file(dirname: str):
    return os.path.join(dirname, "det_metrics.json")


def get_detector_metric_stdev_file(dirname: str):
    return os.path.join(dirname, "det_metrics_stdev.json")


def get_updater_metric_file(dirname: str):
    return os.path.join(dirname, "upd_metrics.json")


def get_updater_metric_stdev_file(dirname: str):
    return os.path.join(dirname, "upd_metrics_stdev.json")


def get_ts_metric_file(dirname: str):
    return os.path.join(dirname, "ts_metrics.json")


def get_ts_metric_stdev_file(dirname: str):
    return os.path.join(dirname, "ts_metrics_stdev.json")
