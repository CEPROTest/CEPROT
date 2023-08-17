# CEPROT

Our model is divided into two stages, identification and generation. And we provide our model and three other baseline models here.

## SITAR
SITAR consists of two processes, extracting the features of the method under the Parse folder and classifying them through four other classification models.

## KNN

You can run it in the order we listed the files.

## NMT

Due to the different data formats, we provide a data conversion script, you can convert the data in (CEPROT/Updating/NMT/convert_data.py). Also, since the NMT generation is concatenated with spaces, we also provide a script to convert the result, which is here(code/updating/nmt/convert_readable.py).


## Our Model

1. To run our indentification model, first you have to configure the file (configs.py). And  run with this command `python run_clone.py --do_train --do_eval --do_test`. During the evaluation process, TPC_id and FPC_id will be generated, please record them, which will be used in the second stage. Of course, you can also run these two phases separately.
2. During the update phase, you can run with `python train_codet5_edit_focal.py`. The premise is that you must configure your own path in the (codet5_edit_focal.py).

#### Note
CodeBleu scores for all generative models can be evaluated using the CEPROT/CodeBLEU/calc_code_bleu.py.

