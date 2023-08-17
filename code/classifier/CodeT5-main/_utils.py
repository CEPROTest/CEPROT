import json
import jsonlines
from numpy import source

def readjson_all(filename):
    d = []
    with open(filename, mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
    for i in dicts:
        d.append(i)
    return d

def readjs(filename):
    l=[]
    with open(filename, "r+", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            l.append(item)
        return l

def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        edit_seq = example.edit_seq
        focal_src = example.focal_src
        focal_tgt = example.focal_tgt
        test_src = example.test_src


    edit_lst = []
    for i in edit_seq:
        edit_lst.extend(i)
    edit_lst = edit_lst[:args.max_source_length]
    e_tmp = []
    for i in edit_lst:
        e_tmp.append(str(i))
    edit_lst = ' '.join(e_tmp)

    source1_ids = tokenizer.encode(edit_lst, max_length=args.max_source_length, padding='max_length', truncation=True)

    source2_ids = tokenizer.encode(focal_src, max_length=args.max_source_length, padding='max_length', truncation=True)

    source3_ids = tokenizer.encode(focal_tgt, max_length=args.max_source_length, padding='max_length', truncation=True)

    source4_ids = tokenizer.encode(test_src, max_length=args.max_source_length, padding='max_length', truncation=True)

    source_ids = source1_ids + source2_ids + source3_ids + source4_ids


    if stage == 'test':
        target_ids = []
    else:
        if args.task in ['defect', 'clone']:
            target_str = example.target
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        else:
            target_str = example.target
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    source_edit = example.edit_seq
    source1 = example.focal_src
    source2 = example.focal_tgt
    source3 = example.test_src
    edit_lst = []
    for i in source_edit:
        edit_lst.extend(i)
    edit_lst = edit_lst[:args.max_source_length]
    edit_lst = ' '.join(edit_lst)
    code_edit = tokenizer.encode(edit_lst, max_length=args.max_source_length, padding='max_length', truncation=True)
    code1 = tokenizer.encode(source1, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(source2, max_length=args.max_source_length, padding='max_length', truncation=True)
    code3 = tokenizer.encode(source3, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids =  code_edit + code1 + code2 + code3
    return CloneInputFeatures(example_index, source_ids, example.target)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    source_edit = example.edit_seq
    source1 = example.focal_src
    source2 = example.focal_tgt
    source3 = example.test_src
    edit_lst = []
    for i in source_edit:
        edit_lst.extend(i)
    edit_lst = edit_lst[:args.max_source_length]
    edit_lst = ' '.join(edit_lst)
    code1 = tokenizer.encode(source1, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(source2, max_length=args.max_source_length, padding='max_length', truncation=True)
    code3 = tokenizer.encode(source3, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids =  code1 + code2 + code3
    return DefectInputFeatures(example_index, source_ids, example.target)

class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = int(label)



class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url



class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 edit_seq,
                 focal_src,
                 focal_tgt,
                 test_src,
                 target,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.edit_seq = edit_seq
        self.focal_src = focal_src
        self.focal_tgt = focal_tgt
        self.test_src = test_src
        self.target = target
        self.task = task
        self.sub_task = sub_task




class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 edit_seq,
                 focal_src,
                 focal_tgt,
                 test_src,
                 target,
                 ):
        self.edit_seq = edit_seq
        self.focal_src = focal_src
        self.focal_tgt = focal_tgt
        self.test_src = test_src
        self.target = int(target)


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    data = readjs(filename)
    examples = []

    idx = 0
    for i in range(len(data)):
        examples.append(
            Example(
                idx=idx,
                edit_seq=data[i]['edit_seq'],
                focal_src=data[i]['focal_src'],
                focal_tgt=data[i]['focal_tgt'],
                test_src=data[i]['test_src'],
                target=data[i]['test_tgt']
            )
        )
        idx += 1
        if idx == data_num:
            break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples




def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    import jsonlines
    data = []
    idx = 0
    with open(filename, "r+", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            # print(item)
            # item focal_src focal_tgt test_src label
            data.append(Example(idx,item['edit_seq'],item['focal_src'], item['focal_tgt'], item['test_src'], int(item['label'])))
            idx += 1
            if idx == data_num:
                break
    return data



def read_clone_examples(filename, data_num):
    import jsonlines
    data = []
    idx = 0
    samples = readjson_all(filename)
    for item in samples:
        data.append(CloneExample(item['edit_seq'],repr(item['focal_src'])[1:-1], repr(item['focal_tgt'])[1:-1], repr(item['test_src'])[1:-1], item['label']))
        idx += 1
        if idx == data_num:
            break
    return data
