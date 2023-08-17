import time
import argparse
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
from turtle import hideturtle
from docopt import docopt
import logging
from model.codet5_edit import Code2Code
from model.codet5_edit import Config
from model.utils import readjson, loadpkl
import pickle
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

from model.codet5_edit import GenDataset, read_examples
import torch
from torch import nn
import os
import multiprocessing
import random
from bleu import _bleu, compute_sentence_level_blue


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cofig = Config()
codet5_tokenizer = RobertaTokenizer.from_pretrained(cofig.decoder_path)
codet5_model = T5ForConditionalGeneration.from_pretrained(cofig.decoder_path).to(device)
model = Code2Code(decoder_model=codet5_model,t5_tokenizer=codet5_tokenizer,config=cofig).to(device)

if cofig.do_train:
    # load train data
    train_examples = read_examples(cofig.train_filename)
    logger.info(" The total sample == %d", len(train_examples))
    train_dataset = GenDataset(cofig, train_examples)
    train_sampler = SequentialSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cofig.batch_size,num_workers=4)
    len_dataset = len(train_examples)
    t_total = (len_dataset // cofig.batch_size) * cofig.num_epochs if len_dataset % cofig.batch_size == 0 else (len_dataset // cofig.batch_size + 1) * cofig.num_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': cofig.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cofig.learning_rate, eps=cofig.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=int(t_total*0.1),
                                                        num_training_steps=t_total)
    dev_dataset={}
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", cofig.batch_size)#cofig.batch_size
    logger.info("  Num epoch = %d", cofig.num_epochs)
    nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6
    best_eval_loss = 1e6
    model.train()
    for epoch in range(cofig.num_epochs):
        print('Epoch {0}'.format(epoch))
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        for batch in bar:
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch
            loss,_ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,target_mask=target_mask)
         
            if cofig.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            tr_loss += loss.item()
            nb_tr_steps += 1
            train_loss=round(tr_loss/(nb_tr_steps+1),4)
            bar.set_description("epoch {} loss {}".format(epoch,train_loss))
            print('loss: ',train_loss)
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()
            if (nb_tr_steps + 1) % cofig.gradient_accumulation_steps == 0:
                        #Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        if cofig.do_eval:
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0                     
            eval_flag=False    
            if 'dev_loss' in dev_dataset:
                eval_examples,eval_data=dev_dataset['dev_loss']
            else:
                eval_examples = read_examples(cofig.dev_filename)
                eval_data = GenDataset(cofig,eval_examples)
                dev_dataset['dev_loss']=eval_examples,eval_data
            eval_sampler = SequentialSampler(eval_data) 
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=cofig.batch_size,num_workers=4)
            logger.info("\n***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", cofig.eval_batch_size)

            #Start Evaling model
            model.eval()
            eval_loss,step_num = 0,1
            for batch in eval_dataloader:   
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,target_ids,target_mask = batch
                with torch.no_grad():
                    loss,_ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,target_mask=target_mask)
                eval_loss = loss.item()
                # model.train()
                eval_loss = eval_loss / step_num
                step_num += 1
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                            'global_step': global_step+1,
                            'train_loss': round(train_loss,5),
                            'eval_loss': round(eval_loss,5)}
                for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)
            # =======================================================
            #save last checkpoint
            last_output_dir = os.path.join(cofig.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

            if eval_loss<best_eval_loss:
                logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                logger.info("  "+"*"*20)
                best_eva_loss=eval_loss
                # Save best checkpoint for best ppl
                output_dir = os.path.join(cofig.output_dir, 'checkpoint-best-ppl')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

            model.eval()
            best_bleu = -100 
            #Calculate bleu  
            if 'dev_loss' in dev_dataset:
                eval_examples,eval_data=dev_dataset['dev_loss']
            else:
                eval_examples = read_examples(cofig.dev_filename)[:100]
                eval_data = GenDataset(cofig,eval_examples)
                dev_dataset['dev_loss']=eval_examples,eval_data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=cofig.batch_size,num_workers=4)
            
            p=[]
            bar = tqdm(eval_dataloader,total=len(eval_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,target_ids,target_mask = batch
                with torch.no_grad():
                    batch_text = model(source_ids=source_ids, source_mask=source_mask, target_ids=None, target_mask=None)
                for text in batch_text:
                    p.append(text)

            print('-=-='*40)
            model.train()
            accs=[]
            predictions = []
            with open(os.path.join(cofig.output_dir,"dev_epoch{0}.output".format(epoch)),'w') as f, open(os.path.join(cofig.output_dir,"dev_epoch{0}.gold".format(epoch)),'w') as f1:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(ref)
                    f.write(ref+'\n')
                    f1.write(gold.test_tgt+'\n')     
                    accs.append(ref==gold.test_tgt)
            sentence_bleu = compute_sentence_level_blue(os.path.join(cofig.output_dir, "dev_epoch{0}.output".format(epoch)), os.path.join(cofig.output_dir, "dev_epoch{0}.gold".format(epoch)) )
            dev_bleu=round(_bleu(os.path.join(cofig.output_dir, "dev_epoch{0}.output".format(epoch)), os.path.join(cofig.output_dir, "dev_epoch{0}.gold".format(epoch))),2)
            with open(os.path.join(cofig.output_dir,"dev_epoch{0}.sentence_bleu".format(epoch)),'w') as f:
                for i in range(len(sentence_bleu)):
                    f.write(str(i)+' '+ str(sentence_bleu[i]) + '\n')
            xmatch=round(np.mean(accs)*100,4)
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
            logger.info("  "+"*"*20)    
            if dev_bleu+xmatch>best_bleu:
                logger.info("  Best BLEU+xMatch:%s",dev_bleu+xmatch)
                logger.info("  "+"*"*20)
                best_bleu=dev_bleu+xmatch
                # Save best checkpoint for best bleu
                output_dir = os.path.join(cofig.output_dir, 'checkpoint-best-bleu')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("  %s = %s "%("Best bleu-4",str(best_bleu)))
        
if cofig.do_test:
    model = Code2Code(codet5_model, codet5_tokenizer, cofig).to(device)
    model.load_state_dict(torch.load(cofig.output_dir + 'checkpoint-last/pytorch_model.bin'))
    model.eval()
    files=[]
    if cofig.test_filename is not None:
        files.append(cofig.test_filename)
    best_bleu = -10
    for idx,file in enumerate(files):   
        logger.info("Test file: {}".format(file))
        test_examples = read_examples(file)
        test_data = GenDataset(cofig,test_examples)

        # Calculate bleu
        test_sampler = SequentialSampler(test_data) 
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=cofig.batch_size,num_workers=4)
        model.eval()
        p=[]
        bar = tqdm(test_dataloader,total=len(test_dataloader))
        i = 0
        for batch in bar:
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch
            with torch.no_grad():
                preds = model(source_ids=source_ids, source_mask=source_mask, target_ids=None, target_mask=None)
                for text in preds:
                    p.append(text)
            accs=[]
        ppp = []
        idx = 0
        with open(os.path.join(cofig.output_dir,"fpc{0}.output".format(i)),'w') as f, open(os.path.join(cofig.output_dir,"fpc{0}.gold".format(i)),'w') as f1:
            for ref,gold in zip(p,test_examples):
                f.write(ref+'\n')
                f1.write(gold.test_tgt+'\n')     
                accs.append(ref==gold.test_tgt)
                if ref == gold.test_tgt:
                    ppp.append(idx)
                idx += 1
        print(ppp)
        dev_bleu=round(_bleu(os.path.join(cofig.output_dir, "fpc{0}.gold".format(i)), os.path.join(cofig.output_dir, "fpc{0}.output".format(i))),2)
        xmatch=round(np.mean(accs)*100,4)
        logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
        logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
        logger.info("  "+"*"*20)    
        logger.info("  Best BLEU+xMatch:%s",dev_bleu+xmatch)
        logger.info("  "+"*"*20)
