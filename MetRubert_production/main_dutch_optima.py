#?##########
#* IMPORTS #
#?##########

import os
import sys
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import optuna

from tqdm import tqdm, trange
from collections import OrderedDict
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, logging

#! Imports from other python file of this module
from utils import Config, Logger, make_log_dir
from modeling import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification_SPV,
    AutoModelForSequenceClassification_MIP,
    AutoModelForSequenceClassification_SPV_MIP,
    AutoModelForSequenceClassification_SPV_MIP_optima,
    AutoModelForSequenceClassification_SPV_MIP_optima_drop,
    AutoModelForSequenceClassification_SPV_MIP_optima_manual
)
from run_classifier_dataset_utils import processors, output_modes, compute_metrics
from data_loader import load_train_data, load_test_data

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"

#?###########
#* SETTINGS #
#?###########

#!warnings and error settings
amount_of_info = 4 #1 = all information, 2 = errors, warnings, basic info, 3 = error warnings, 4 = errors

#?output settings
print_model = False
cuda_output = False
training_output = True
best_output = True

#*hyperparameter optimizer settings
optunamode = True
optuna_trials = 1000
optuna_plot = False

optuna_tweak_seed = False
optuna_tweak_hidden_layers = False
optuna_tweak_drop_ratio = False
optuna_tweak_learning_rate = True

#* manual hpo settings
manualmode = False

#?############
#* main code #
#?############

#First interprete the logging settings
if amount_of_info == 1:
    logging.set_verbosity_debug
elif amount_of_info == 2:
    logging.set_verbosity_info
elif amount_of_info == 3:
    logging.set_verbosity_warning
elif amount_of_info == 4:
    logging.set_verbosity_error

out = open("outputs.txt", "w")
best_f1_found = 0

def main():
    if amount_of_info == 1:
        logging.set_verbosity_debug
    elif amount_of_info == 2:
        logging.set_verbosity_info
    elif amount_of_info == 3:
        logging.set_verbosity_warning
    elif amount_of_info == 4:
        logging.set_verbosity_error

    if optunamode == True:
        #We create an optuna study with as goal to maximize the number we put into it.
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=optuna_trials)

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    elif manualmode == True:
        objective_manual(768, 0.2, 7.9e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.85e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.86e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.87e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.88e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.89e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.891e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.891e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.892e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.893e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.894e-5, 3, 2, 42, 32)    
        objective_manual(768, 0.2, 7.895e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.896e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.897e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.898e-5, 3, 2, 42, 32)  
        objective_manual(768, 0.2, 7.899e-5, 3, 2, 42, 32)
        objective_manual(768, 0.2, 7.901e-5, 3, 2, 42, 32)        
        objective_manual(768, 0.2, 7.902e-5, 3, 2, 42, 32)  
        objective_manual(768, 0.2, 7.903e-5, 3, 2, 42, 32)  
        objective_manual(768, 0.2, 7.904e-5, 3, 2, 42, 32)  
        objective_manual(768, 0.2, 7.905e-5, 3, 2, 42, 32)  
        objective_manual(768, 0.2, 7.906e-5, 3, 2, 42, 32)  
        objective_manual(768, 0.2, 7.907e-5, 3, 2, 42, 32)  
        objective_manual(768, 0.2, 7.908e-5, 3, 2, 42, 32)  
        objective_manual(768, 0.2, 7.909e-5, 3, 2, 42, 32)  
        objective_manual(768, 0.2, 7.91e-5, 3, 2, 42, 32)      
        objective_manual(768, 0.2, 7.92e-5, 3, 2, 42, 32)   
        objective_manual(768, 0.2, 7.93e-5, 3, 2, 42, 32)   
        objective_manual(768, 0.2, 7.94e-5, 3, 2, 42, 32)   
        objective_manual(768, 0.2, 7.95e-5, 3, 2, 42, 32)   
        objective_manual(768, 0.2, 7.96e-5, 3, 2, 42, 32)   
        objective_manual(768, 0.2, 7.97e-5, 3, 2, 42, 32)   
        objective_manual(768, 0.2, 7.98e-5, 3, 2, 42, 32)   
        objective_manual(768, 0.2, 7.99e-5, 3, 2, 42, 32)
        objective_manual(1096, 0.2, 8.220e-5, 3, 2, 42, 32)
        objective_manual(1096, 0.2, 8.225e-5, 3, 2, 42, 32)
        objective_manual(1096, 0.2, 8.230e-5, 3, 2, 42, 32)
        objective_manual(1096, 0.2, 8.215e-5, 3, 2, 42, 32)
        objective_manual(1096, 0.2, 8.235e-5, 3, 2, 42, 32)
        objective_manual(1096, 0.2, 8.226e-5, 3, 2, 42, 32)
        objective_manual(1096, 0.2, 8.227e-5, 3, 2, 42, 32)
        objective_manual(1096, 0.2, 8.224e-5, 3, 2, 42, 32)
        objective_manual(1096, 0.2, 8.223e-5, 3, 2, 42, 32)
        objective_manual(461, 0.2, 8.637e-5, 3, 2, 42, 32)
        objective_manual(461, 0.2, 8.64e-5, 3, 2, 42, 32)
        objective_manual(461, 0.2, 8.62e-5, 3, 2, 42, 32)
        objective_manual(461, 0.2, 8.635e-5, 3, 2, 42, 32)
        objective_manual(461, 0.2, 8.6385e-5, 3, 2, 42, 32)
        objective_manual(461, 0.2, 8.64e-5, 3, 2, 42, 32)
        #We get the option to manually run certain data 

    if optuna_plot == True:
        optuna.visualization.plot_optimization_history(study)
        optuna.visualization.plot_slice(study)
        optuna.visualization.plot_contour(study, params=['n_estimators', 'max_depth'])




#?############
#* FUNCTIONS #
#?############

def objective_manual(hid_layer, drop_ratio, lear_rate, num_epochs, warmup_epochs, rand_seed, ba_size):

    #* read configuration into config via /utils/Config.py
    config = Config(main_conf_path="./")

    # apply system arguments if exist
    argv = sys.argv[1:]
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = " ".join(sys.argv[1:]).split(" ")
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip("-")
            cmd_arg[arg_name] = arg_value
        config.update_params(cmd_arg)

    args = config

    if (print_model == True):
        print(args.__dict__)

    # logger
    if "saves" in args.bert_model:
        log_dir = args.bert_model
        logger = Logger(log_dir)
        config = Config(main_conf_path=log_dir)
        old_args = copy.deepcopy(args)
        args.__dict__.update(config.__dict__)

        args.bert_model = old_args.bert_model
        args.do_train = old_args.do_train
        args.data_dir = old_args.data_dir
        args.task_name = old_args.task_name

        # apply system arguments if exist
        argv = sys.argv[1:]
        if len(argv) > 0:
            cmd_arg = OrderedDict()
            argvs = " ".join(sys.argv[1:]).split(" ")
            for i in range(0, len(argvs), 2):
                arg_name, arg_value = argvs[i], argvs[i + 1]
                arg_name = arg_name.strip("-")
                cmd_arg[arg_name] = arg_value
            config.update_params(cmd_arg)
    else:

        #? Setup logger if this is the first run.
        if not os.path.exists("saves"):
            os.mkdir("saves")
        log_dir = make_log_dir(os.path.join("saves", args.bert_model))
        logger = Logger(log_dir)
        config.save(log_dir)
    args.log_dir = log_dir

    #? set CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    #* Display user what version we are using
    if cuda_output == True and torch.cuda.is_available() and not args.no_cuda:
        print("Using CUDA")
        logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))
    elif cuda_output == True:
        print("USING CPU")
        logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))

    #* SEED TWEEKING
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(rand_seed)

    #? get dataset and processor
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)
    
    #######
    #DUTCH#
    #######
    #* build tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base", do_lower_case=args.do_lower_case)
    model = load_pretrained_model_manual(hid_layer, drop_ratio, args)

    #!########## Training ###########
    if args.do_train and args.task_name == "vua":
        train_dataloader = load_train_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        best_result = train_me_manual(
	        lear_rate,
            num_epochs,
            warmup_epochs,
            ba_size,
            args,
            logger,
            model,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )
        global best_f1_found
        if best_result > best_f1_found:
            best_f1_found = best_result
            out.write("NEW BEST RESULT \n")
        out.write("RESULT: " + str(best_result) + " hid layer: "+ str(hid_layer) + " drop ratio: " + str(drop_ratio) +  " lear_rate: " + str(lear_rate) + " num epochs:" + str(
            num_epochs) + " warmup epochs: " + str(warmup_epochs) + " random seed:" + str(rand_seed) + " ba_size: " + str(ba_size) + "\n")
        return best_result

    if best_output == True:
        logger.info(f"Saved to {logger.log_dir}")


def objective(trial):

    #* read configuration into config via /utils/Config.py
    config = Config(main_conf_path="./")

    # apply system arguments if exist
    argv = sys.argv[1:]
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = " ".join(sys.argv[1:]).split(" ")
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip("-")
            cmd_arg[arg_name] = arg_value
        config.update_params(cmd_arg)

    args = config

    if (print_model == True):
        print(args.__dict__)

    # logger
    if "saves" in args.bert_model:
        log_dir = args.bert_model
        logger = Logger(log_dir)
        config = Config(main_conf_path=log_dir)
        old_args = copy.deepcopy(args)
        args.__dict__.update(config.__dict__)

        args.bert_model = old_args.bert_model
        args.do_train = old_args.do_train
        args.data_dir = old_args.data_dir
        args.task_name = old_args.task_name

        # apply system arguments if exist
        argv = sys.argv[1:]
        if len(argv) > 0:
            cmd_arg = OrderedDict()
            argvs = " ".join(sys.argv[1:]).split(" ")
            for i in range(0, len(argvs), 2):
                arg_name, arg_value = argvs[i], argvs[i + 1]
                arg_name = arg_name.strip("-")
                cmd_arg[arg_name] = arg_value
            config.update_params(cmd_arg)
    else:

        #? Setup logger if this is the first run.
        if not os.path.exists("saves"):
            os.mkdir("saves")
        log_dir = make_log_dir(os.path.join("saves", args.bert_model))
        logger = Logger(log_dir)
        config.save(log_dir)
    args.log_dir = log_dir

    #? set CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    #* Display user what version we are using
    if cuda_output == True and torch.cuda.is_available() and not args.no_cuda:
        print("Using CUDA")
        logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))
    elif cuda_output == True:
        print("USING CPU")
        logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))

    #* OPTUNA SEED TWEEKING
    if optuna_tweak_seed == True:
        optuna_seed = trial.suggest_int('optuna_seed', 0, 100)
        random.seed(optuna_seed)
        np.random.seed(optuna_seed)
        torch.manual_seed(optuna_seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(optuna_seed)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    #? get dataset and processor
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)
    
    #######
    #DUTCH#
    #######
    #* build tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base", do_lower_case=args.do_lower_case)

    if optunamode == True:
        model = load_pretrained_model(trial, args)

    #!########## Training ###########
    if args.do_train and args.task_name == "vua":
        train_dataloader = load_train_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        best_result = train_me(
	    trial,
            args,
            logger,
            model,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )
        return best_result
    if best_output == True:
        logger.info(f"Saved to {logger.log_dir}")

###################################
#Joost optuna variant of run_train#
###################################
    
def train_me(
    trial,
    args,
    logger,
    model,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_mode,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch

    #? Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    #optuna edit by joost
    if optuna_tweak_learning_rate == True:
        lear_rate = trial.suggest_float('lear_rate', 7e-5, 15e-5)
        optimizer = AdamW(optimizer_grouped_parameters, lr=lear_rate)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_schedule != False or args.lr_schedule.lower() != "none":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )

    if training_output == True:
        logger.info("***** Running training *****")
        logger.info(f"  Batch size = {args.train_batch_size}")
        logger.info(f"  Num steps = { num_train_optimization_steps}")

    #? Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT"]:
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    label_ids,
                    input_ids_2,
                    input_mask_2,
                    segment_ids_2,
                ) = batch
            else:
                input_ids, input_mask, segment_ids, label_ids = batch

            #* compute loss values
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

            #* average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule != False or args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()

        cur_lr = optimizer.param_groups[0]["lr"]
        if training_output == True:
            logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        #? evaluate
        if args.do_eval:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name)

            #? update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    save_model(args, model, tokenizer)
            if args.task_name == "vua":
                save_model(args, model, tokenizer)

    if best_output == True:
        logger.info(f"-----Best Result-----")
        for key in sorted(max_result.keys()):
            logger.info(f"  {key} = {str(max_result[key])}")

    #We return the maximum f1 score in this example, because that is what we want to optimize with optuna
    return max_val_f1

def train_me_manual(
    lear_rate_manual,
    train_epoch_manual,
    warmup_epoch_manual,
    train_batch_size_manual,
    args,
    logger,
    model,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_mode,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * train_epoch_manual

    #? Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=lear_rate_manual)
    if args.lr_schedule != False or args.lr_schedule.lower() != "none":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_epoch_manual * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )

    if training_output == True:
        logger.info("***** Running training *****")
        logger.info(f"  Batch size = {train_batch_size_manual}")
        logger.info(f"  Num steps = { num_train_optimization_steps}")

    #? Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    for epoch in trange(int(train_epoch_manual), desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT"]:
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    label_ids,
                    input_ids_2,
                    input_mask_2,
                    segment_ids_2,
                ) = batch
            else:
                input_ids, input_mask, segment_ids, label_ids = batch

            #* compute loss values
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

            #* average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule != False or args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()

        cur_lr = optimizer.param_groups[0]["lr"]
        if training_output == True:
            logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        #? evaluate
        if args.do_eval:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name)

            #? update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    save_model(args, model, tokenizer)
            if args.task_name == "vua":
                save_model(args, model, tokenizer)

    if best_output == True:
        logger.info(f"-----Best Result-----")
        for key in sorted(max_result.keys()):
            logger.info(f"  {key} = {str(max_result[key])}")

    #We return the maximum f1 score in this example, because that is what we want to optimize with optuna
    return max_val_f1

def run_train(
    args,
    logger,
    model,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_mode,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch

    #? Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_schedule != False or args.lr_schedule.lower() != "none":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )

    if training_output == True:
        logger.info("***** Running training *****")
        logger.info(f"  Batch size = {args.train_batch_size}")
        logger.info(f"  Num steps = { num_train_optimization_steps}")

    #? Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT"]:
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    label_ids,
                    input_ids_2,
                    input_mask_2,
                    segment_ids_2,
                ) = batch
            else:
                input_ids, input_mask, segment_ids, label_ids = batch

            #* compute loss values
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

            #* average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule != False or args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        #? evaluate
        if args.do_eval:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name)

            #? update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    save_model(args, model, tokenizer)
            if args.task_name == "vua":
                save_model(args, model, tokenizer)

    logger.info(f"-----Best Result-----")
    for key in sorted(max_result.keys()):
        logger.info(f"  {key} = {str(max_result[key])}")

    return model, max_result


def run_eval(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=False):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    pred_guids = []
    out_label_ids = None

    for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)

        if args.model_type in ["MELBERT_MIP", "MELBERT"]:
            (
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                idx,
                input_ids_2,
                input_mask_2,
                segment_ids_2,
            ) = eval_batch
        else:
            input_ids, input_mask, segment_ids, label_ids, idx = eval_batch

        with torch.no_grad():
            # compute loss values
            if args.model_type in ["BERT_BASE", "BERT_SEQ", "MELBERT_SPV"]:
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    #? compute metrics
    result = compute_metrics(preds, out_label_ids)

    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

    if return_preds:
        return preds
    return result


def load_pretrained_model(trial, args):
    #? Pretrained Model
    bert = AutoModel.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    config = bert.config
    config.type_vocab_size = 4
    if "albert" in args.bert_model:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )
    else:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
    bert._init_weights(bert.embeddings.token_type_embeddings)

    # Additional Layers
    if args.model_type in ["BERT_BASE"]:
        model = AutoModelForSequenceClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "BERT_SEQ":
        model = AutoModelForTokenClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_SPV":
        model = AutoModelForSequenceClassification_SPV(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_MIP":
        model = AutoModelForSequenceClassification_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )

    #! PROBABLY INTERESTING FOR CONVERSION TO DUTCH
    if args.model_type == "MELBERT":
        if optuna_tweak_hidden_layers == True:
            model = AutoModelForSequenceClassification_SPV_MIP_optima(
                trial, args=args, Model=bert, config=config, num_labels=args.num_labels
            )
        elif optuna_tweak_drop_ratio == True:
            model = AutoModelForSequenceClassification_SPV_MIP_optima_drop(
                trial, args=args, Model=bert, config=config, num_labels=args.num_labels
            )
        else:
            model = AutoModelForSequenceClassification_SPV_MIP(
                args=args, Model=bert, config=config, num_labels=args.num_labels
            )

    model.to(args.device)
    if args.n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)
    return model

def load_pretrained_model_manual(am_hidden, am_drop, args):
    #? Pretrained Model
    bert = AutoModel.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    config = bert.config
    config.type_vocab_size = 4
    if "albert" in args.bert_model:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )
    else:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
    bert._init_weights(bert.embeddings.token_type_embeddings)

    # Additional Layers
    if args.model_type in ["BERT_BASE"]:
        model = AutoModelForSequenceClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "BERT_SEQ":
        model = AutoModelForTokenClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_SPV":
        model = AutoModelForSequenceClassification_SPV(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_MIP":
        model = AutoModelForSequenceClassification_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )

    #! PROBABLY INTERESTING FOR CONVERSION TO DUTCH
    if args.model_type == "MELBERT":
        model = AutoModelForSequenceClassification_SPV_MIP_optima_manual(
            am_hidden, am_drop, args=args, Model=bert, config=config, num_labels=args.num_labels)

    model.to(args.device)
    if args.n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)
    return model

def save_model(args, model, tokenizer):
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.log_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.log_dir)

    # Good practice: save your training arguments together with the trained model
    output_args_file = os.path.join(args.log_dir, ARGS_NAME)
    torch.save(args, output_args_file)


def load_trained_model(args, model, tokenizer):
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)

    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(output_model_file))
    else:
        model.load_state_dict(torch.load(output_model_file))

    return model

if __name__ == "__main__":
    main()

