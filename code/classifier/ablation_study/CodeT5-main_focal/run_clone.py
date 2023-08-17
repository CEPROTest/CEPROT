from __future__ import absolute_import
from cProfile import label
import os

from psutil import cpu_count
from models import CloneModel
import logging
import argparse
import math
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc  
import time

from configs import add_args, set_seed
from utils import get_filenames, get_elapse_time, load_and_cache_clone_data, flat_scores
from models import get_model_size

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

cpu_cont = 12

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def my_int(x):
    return int(x)


def evaluate(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)

    output = open('pre_score.txt','w+')
    for i in range(len(logits)):
        for j in range(len(logits[i])):
            output.write(str(logits[i][j]))
            output.write(' ')   
        output.write('\n')      
    output.close()

    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    for threshold in np.linspace(0,1,11):
        best_threshold = threshold
        y_preds = logits[:, 1] > best_threshold

        y_preds = list(map(my_int,y_preds))
        recall = flat_scores(y_preds, y_trues, 'recall')
        precision = flat_scores(y_preds, y_trues, 'precision')
        f1 = flat_scores(y_preds, y_trues, 'f1')
        acc = flat_scores(y_preds,y_trues,'acc')
        tpr = flat_scores(y_preds, y_trues, 'tpr')
        fpr = flat_scores(y_preds, y_trues, 'fpr')
        TP, FP, TN, FN = flat_scores(y_preds, y_trues, 'detail')
        assert len(y_trues) == len(y_preds)
        tpc_id = []
        fpc_id = []
        for i in range(len(y_trues)):
            if y_trues[i] == y_preds[i] and y_trues[i] == 1:
                tpc_id.append(i)
            if y_trues[i] == 0 and y_preds[i] == 1:
                fpc_id.append(i)


        roc_auc = auc([0,fpr,1],[0,tpr,1])     

        result = {
            'acc':acc,
            "eval_recall": recall,
            "eval_precision": precision,
            "eval_f1": f1,
            "fpr":fpr,
            'tpr':tpr,
            "auc":roc_auc,
            "TP":TP,
            "FP":FP,
            "TN":TN,
            "FN":FN,
            'tpc':str(tpc_id),
            'fpc':str(fpc_id),
            "eval_threshold": best_threshold,
        }

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            # logger.info("  %s = %s", key, str(round(result[key], 4)))
            logger.info("  %s = %s",key,result[key])
        logger.info("  " + "*" * 20)

    return result

def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    model = CloneModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    pool = multiprocessing.Pool(cpu_cont)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')


    if args.do_train:
        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_clone_data(args, args.train_filename, pool, tokenizer, 'train',
                                                               is_sample=False)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        save_steps = 2500

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        global_step, best_f1 = 0, 0
        not_f1_inc_cnt = 0
        is_early_stop = False
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()

            for step, batch in enumerate(bar):

                batch = tuple(t.to(device) for t in batch)
                source_ids, labels = batch

                loss, logits = model(source_ids, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                if (step + 1) % save_steps == 0 and args.do_eval:
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    eval_examples, eval_data = load_and_cache_clone_data(args, args.dev_filename, pool, tokenizer,
                                                                        'valid', is_sample=True)

                    result = evaluate(args, model, eval_examples, eval_data)
                    eval_f1 = result['eval_f1']

                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_f1', round(eval_f1, 4), cur_epoch)

                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    cur_output_dir = os.path.join(args.output_dir,'epoch'+str(cur_epoch))
                    if not os.path.exists(cur_output_dir):
                        os.makedirs(cur_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(cur_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the current epoch model into %s", output_model_file)


                    if True or args.data_num == -1 and args.save_last_checkpoints:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s", output_model_file)

                    if eval_f1 > best_f1:
                        not_f1_inc_cnt = 0
                        logger.info("  Best f1: %s", round(eval_f1, 4))
                        logger.info("  " + "*" * 20)
                        fa.write("[%d] Best f1 changed into %.4f\n" % (cur_epoch, round(eval_f1, 4)))
                        best_f1 = eval_f1
                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or True:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best ppl model into %s", output_model_file)
        
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['last']:
            file = os.path.join(args.output_dir, 'pytorch_model.bin')
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))

            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            eval_examples, eval_data = load_and_cache_clone_data(args, args.test_filename, pool, tokenizer, 'test',
                                                                 False)

            result = evaluate(args, model, eval_examples, eval_data, write_to_pred=True)
            logger.info("  test_f1=%.4f", result['eval_f1'])
            logger.info("  test_prec=%.4f", result['eval_precision'])
            logger.info("  test_rec=%.4f", result['eval_recall'])
            logger.info("  " + "*" * 20)

            fa.write("[%s] test-f1: %.4f, precision: %.4f, recall: %.4f\n" % (
                criteria, result['eval_f1'], result['eval_precision'], result['eval_recall']))
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write("[%s] f1: %.4f, precision: %.4f, recall: %.4f\n\n" % (
                        criteria, result['eval_f1'], result['eval_precision'], result['eval_recall']))
    fa.close()

if __name__ == "__main__":
    main()


