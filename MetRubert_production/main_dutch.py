
#?##########
#* IMPORTS #
#?##########

import os
import sys
import random
import copy
import numpy as np
import pickle

import torch
import torch.nn as nn

from colorama import Fore
from tqdm import tqdm, trange
from collections import OrderedDict
from transformers import AutoModel, get_linear_schedule_with_warmup, RobertaTokenizer, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

#! Imports from other python file of this module
from MetRobert_run2.utils import Config, Logger, make_log_dir
from MetRobert_run2. modeling import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification_SPV,
    AutoModelForSequenceClassification_MIP,
    AutoModelForSequenceClassification_SPV_MIP,
)
from MetRobert_run2.run_classifier_dataset_utils import processors, output_modes, compute_metrics
from MetRobert_run2.data_loader import load_train_data, load_test_data, load_dev_data, load_train_data_kf

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"

#?###########
#* SETTINGS #
#?###########

print_model = False
cuda_output = False
save_test_pred = False
save_dev_pred = False
training_output = False

devout = open("output/predictions_dev.txt", "w")
devoutf = open("output/predictions_dev_float.txt", "w")
devouts = open("output/predictions_dev_soft.txt", "w")

#?############
#* FUNCTIONS #
#?############


def main(dev_dir):

    # * INFOGRAPHIC OUTPUT
    print(Fore.LIGHTGREEN_EX + "   *                        (           ")
    print(" (  `         (    (        )\ )  *   ) ")
    print(" )\))(     (  )\ ( )\  (   (()/(` )  /( ")
    print("((_)()\   ))\((_))((_) )\   /(_))( )(_))")
    print("(_()((_) /((_)_ ((_)_ ((_) (_)) (_(_()) ")
    print("|  \/  |(_)) | | | _ )| __|| _ \|_   _| ")
    print("| |\/| |/ -_)| | | _ \| _| |   /  | |   ")
    print("|_|  |_|\___||_| |___/|___||_|_\  |_|   ")
    print("")
    print(Fore.WHITE + "             DUTCH EDITION             ")

    # * read configuration into config via /utils/Config.py
    config = Config(main_conf_path="/vol/tensusers4/jgrunwald/metaphorclam/metaphorclam/MetRobert_run2/")

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

        # ? Setup logger if this is the first run.
        if not os.path.exists("saves"):
            os.mkdir("saves")
        log_dir = make_log_dir(os.path.join("saves", args.bert_model))
        logger = Logger(log_dir)
        config.save(log_dir)
    args.log_dir = log_dir

    # ? set CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # * Display user what version we are using
    if cuda_output == True and torch.cuda.is_available() and not args.no_cuda:
        print("Using CUDA")
        logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))
    elif cuda_output == True:
        print("USING CPU")
        logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))

    # * set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # ? get dataset and processor
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    # * build tokenizer and model
    # TODO: CHANGE FOR DUTCH
    tokenizer = RobertaTokenizer.from_pretrained(
        "pdelobelle/robbert-v2-dutch-base", do_lower_case=args.do_lower_case)
    model = load_pretrained_model(args)
    
    #!########## Training ###########
    #! VUA-18 / VUA-20 with bagging
    if args.do_train and args.task_name == "vua" and args.num_bagging:
        train_data, gkf = load_train_data_kf(args, logger, processor, task_name, label_list, tokenizer, output_mode)

        for fold, (train_idx, valid_idx) in enumerate(tqdm(gkf, desc="bagging...")):
            if fold != args.bagging_index:
                continue

            print(f"bagging_index = {args.bagging_index}")

            # Load data
            temp_train_data = TensorDataset(*train_data[train_idx])
            train_sampler = RandomSampler(temp_train_data)
            train_dataloader = DataLoader(temp_train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            # Reset Model
            model = load_pretrained_model(args)
            model, best_result = run_train(args, logger, model, train_dataloader, processor, task_name, label_list, tokenizer, output_mode)

            # Test
            all_guids, eval_dataloader = load_test_data(args, logger, processor, task_name, label_list, tokenizer, output_mode)
            preds = run_eval(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=True)
            with open(os.path.join(args.data_dir, f"seed{args.seed}_preds_{fold}.p"), "wb") as f:
                pickle.dump(preds, f)

            # If train data is VUA20, the model needs to be tested on VUAverb, MOH-X, TroFi as well.
            # You can just adjust the names of data_dir in conditions below for your own data directories.
            # if "VUA20" in args.data_dir:
            #     # Verb
            #     args.data_dir = "data/VUAverb"
            #     all_guids, eval_dataloader = load_test_data(args, logger, processor, task_name, label_list, tokenizer, output_mode)
            #     preds = run_eval(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=True)
            #     with open(os.path.join(args.data_dir, f"seed{args.seed}_preds_{fold}.p"), "wb") as f:
            #         pickle.dump(preds, f)

            logger.info(f"Saved to {logger.log_dir}")
        return                

    #!########## Training ###########
    #! VUA-18 / VUA-20
    if args.do_train and args.task_name == "vua":
        train_dataloader = load_train_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train(
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

    #! TroFi / MOH-X (K-fold)
    # ? The difference is that this model also holds a k (k_fold)
    elif args.do_train and args.task_name == "trofi":
        k_result = []
        for k in tqdm(range(args.kfold), desc="K-fold"):
            model = load_pretrained_model(args)
            train_dataloader = load_train_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            model, best_result = run_train(
                args,
                logger,
                model,
                train_dataloader,
                processor,
                task_name,
                label_list,
                tokenizer,
                output_mode,
                k,
            )
            k_result.append(best_result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Average Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")

    # ? Code that runs when loading trained model
    model = load_trained_model(args, model, tokenizer)
    if args.do_pred:
       all_guids, dev_dataloader = load_dev_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, dev_dir, None)
       predictions = run_dev(args, logger, model, dev_dataloader, all_guids, task_name)

    #!########## Inference ###########
    #! VUA-18 / VUA-20
    # ? This code runs specific kind of tests, like genre and position
    if (args.do_eval or args.do_test) and task_name == "vua" and True == False:
        # if test data is genre or POS tag data
        if ("genre" in args.data_dir) or ("pos" in args.data_dir):
            if "genre" in args.data_dir:
                targets = ["acad", "conv", "fict", "news"]
            elif "pos" in args.data_dir:
                targets = ["adj", "adv", "noun", "verb"]
            orig_data_dir = args.data_dir
            for idx, target in tqdm(enumerate(targets)):
                logger.info(
                    f"====================== Evaluating {target} =====================")
                args.data_dir = os.path.join(orig_data_dir, target)
                all_guids, eval_dataloader = load_test_data(
                    args, logger, processor, task_name, label_list, tokenizer, output_mode
                )
                run_eval(args, logger, model, eval_dataloader,
                         all_guids, task_name)
        else:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode
            )
            run_eval(args, logger, model, eval_dataloader,
                     all_guids, task_name)

    #! TroFi / MOH-X (K-fold)
    elif (args.do_eval or args.do_test) and args.task_name == "trofi" and True == False:
        logger.info(f"***** Evaluating with {args.data_dir}")
        k_result = []
        for k in tqdm(range(10), desc="K-fold"):
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval(args, logger, model,
                              eval_dataloader, all_guids, task_name)
            k_result.append(result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")
    logger.info(f"Saved to {logger.log_dir}")


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

    # ? Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in param_optimizer
                if all(nd not in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if any(nd in n for nd in no_decay)
            ],
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

    # ? Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration",
                                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTGREEN_EX, Fore.RESET))):
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

            # * compute loss values
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor(
                    [1, args.class_weight]).to(args.device))
                loss = loss_fct(
                    logits.view(-1, args.num_labels), label_ids.view(-1))
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
                loss_fct = nn.NLLLoss(weight=torch.Tensor(
                    [1, args.class_weight]).to(args.device))
                loss = loss_fct(
                    logits.view(-1, args.num_labels), label_ids.view(-1))

            # * average loss if on multi-gpu.
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

        # ? evaluate
        if args.do_test:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name, False)

            # ? update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    save_model(args, model, tokenizer)
            if args.task_name == "vua":
                save_model(args, model, tokenizer)

        #! Call code only once!
        if (epoch == args.num_train_epoch-1):
            if (save_test_pred == True):
                # ? Save test predictions option
                predictions = run_eval(
                    args, logger, model, eval_dataloader, all_guids, task_name, True)
                for pred in predictions:
                    out.write(str(pred) + "\n")
            if (save_dev_pred == True and args.do_eval == True):
                all_guids, dev_dataloader = load_dev_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k)
                predictions = run_dev(
                    args, logger, model, dev_dataloader, all_guids, task_name)

    logger.info('-----Best Result-----')
    for key in sorted(max_result.keys()):
        logger.info(f'  {key} = {max_result[key]}')

    return model, max_result

def sigmoid(X):
    return 1/(1+np.exp(-X))

def run_dev(args, logger, model, dev_dataloader, all_guids, task_name):
    #! This function is for now only written for MELBERT models
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    pred_guids = []
    out_label_ids = None

    for dev_batch in dev_dataloader:
        dev_batch = tuple(t.to(args.device) for t in dev_batch)

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
            ) = dev_batch
        else:
            input_ids, input_mask, segment_ids, label_ids, idx = dev_batch
        with torch.no_grad():

            if args.model_type in ["MELBERT_MIP", "MELBERT"]:
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
                tmp_eval_loss = loss_fct(
                    logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if not preds:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

    preds = preds[0]
    predgu = pred_guids[0]

    if(len(preds) != len(pred_guids)):
       print("ERROR:  SIZE OF GUIDS DOES NOT MATCH SIZE OF PREDS")

    predrange = len(preds)

    #? We save our decimal predictions over here.
    predsdec = preds #We save exact numbers
    for i in range(predrange):
        devoutf.write(str(predgu[i]) + "," + str(predsdec[i]) + "\n")

    #* change to 0 or 1 predictions
    preds = np.argmax(preds, axis=1)

    for i in range(predrange):
        devout.write(str(predgu[i]) + "," + str(preds[i]) + "\n")

    predssof = sigmoid(predsdec)

    for i in range(predrange):
        devouts.write(str(predgu[i]) + "," + str(predssof[i]) + "\n")

    devout.close()
    devoutf.close()
    devouts.close()

    return preds


def run_eval(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=False):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    pred_guids = []
    out_label_ids = None

    for eval_batch in tqdm(eval_dataloader, desc="Evaluating2",  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
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
                tmp_eval_loss = loss_fct(
                    logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)
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
                tmp_eval_loss = loss_fct(
                    logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    # ? compute metrics
    result = compute_metrics(preds, out_label_ids)

    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

    if return_preds:
        return preds
    return result


def load_pretrained_model(args):
    # ? Pretrained Model
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
        model = AutoModelForSequenceClassification_SPV_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )

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
