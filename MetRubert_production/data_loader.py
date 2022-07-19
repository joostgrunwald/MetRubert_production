
#?##########
#* IMPORTS #
#?##########

import torch

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from MetRobert_run2.run_classifier_dataset_utils import (
    convert_examples_to_two_features,
    convert_examples_to_features,
    convert_two_examples_to_features,
)

#?############
#* FUNCTIONS #
#?############

def load_train_data(
    args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None
):
    # Prepare data loader
    if task_name == "vua":
        train_examples = processor.get_train_examples(args.data_dir)
    elif task_name == "trofi":
        train_examples = processor.get_train_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    #DEBUG
    #for example in train_examples:
    #   print(example.text_a)
    #   print(example.text_b)
    #   print(example.POS)
    #   make features file
    
    if args.model_type == "BERT_BASE":
        train_features = convert_two_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        train_features = convert_examples_to_features(
            train_examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            output_mode,
            args,
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        train_features = convert_examples_to_two_features(
            train_examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            output_mode,
            args,
        )

    # make features into tensor
    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long
    )
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # add additional features for MELBERT_MIP and MELBERT
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids_2 = torch.tensor(
            [f.input_ids_2 for f in train_features], dtype=torch.long
        )
        all_input_mask_2 = torch.tensor(
            [f.input_mask_2 for f in train_features], dtype=torch.long
        )
        all_segment_ids_2 = torch.tensor(
            [f.segment_ids_2 for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    else:
        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
    train_sampler = RandomSampler(train_data)
    return DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

def load_train_data_kf(
    args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None
):
    # Prepare data loader
    if task_name == "vua":
        train_examples = processor.get_train_examples(args.data_dir)
    elif task_name == "trofi":
        train_examples = processor.get_train_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    # make features file
    if args.model_type == "BERT_BASE":
        train_features = convert_two_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        train_features = convert_examples_to_features(
            train_examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            output_mode,
            args,
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        train_features = convert_examples_to_two_features(
            train_examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            output_mode,
            args,
        )

    # make features into tensor
    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long
    )
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # add additional features for MELBERT_MIP and MELBERT
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids_2 = torch.tensor(
            [f.input_ids_2 for f in train_features], dtype=torch.long
        )
        all_input_mask_2 = torch.tensor(
            [f.input_mask_2 for f in train_features], dtype=torch.long
        )
        all_segment_ids_2 = torch.tensor(
            [f.segment_ids_2 for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    else:
        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
    gkf = StratifiedKFold(n_splits=args.num_bagging).split(
        X=all_input_ids, y=all_label_ids.numpy()
    )
    return train_data, gkf


def load_test_data(
    args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None
):
    if task_name == "vua":
        eval_examples = processor.get_test_examples(args.data_dir)
    elif task_name == "trofi":
        eval_examples = processor.get_test_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    if args.model_type == "BERT_BASE":
        eval_features = convert_two_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        eval_features = convert_examples_to_two_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    logger.info("***** Running evaluation *****")
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long
        )
        all_guids = [f.guid for f in eval_features]
        all_idx = torch.tensor(list(range(len(eval_features))), dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long
        )
        all_input_ids_2 = torch.tensor(
            [f.input_ids_2 for f in eval_features], dtype=torch.long
        )
        all_input_mask_2 = torch.tensor(
            [f.input_mask_2 for f in eval_features], dtype=torch.long
        )
        all_segment_ids_2 = torch.tensor(
            [f.segment_ids_2 for f in eval_features], dtype=torch.long
        )
        eval_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_idx,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    else:
        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long
        )
        all_guids = [f.guid for f in eval_features]
        all_idx = torch.tensor(list(range(len(eval_features))), dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long
        )
        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_idx
        )

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    return all_guids, eval_dataloader


def load_dev_data(
    args, logger, processor, task_name, label_list, tokenizer, output_mode, dev_dir, k=None
):
    if task_name == "vua":
        dev_examples = processor.get_dev_examples(args.data_dir, dev_dir)
    elif task_name == "trofi":
        dev_examples = processor.get_dev_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    if args.model_type == "BERT_BASE":
        dev_features = convert_two_examples_to_features(
            dev_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        dev_features = convert_examples_to_features(
            dev_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        dev_features = convert_examples_to_two_features(
            dev_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    logger.info("***** Running model on dev data *****")
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids = torch.tensor(
            [f.input_ids for f in dev_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in dev_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in dev_features], dtype=torch.long
        )
        all_guids = [f.guid for f in dev_features]
        all_idx = torch.tensor(list(range(len(dev_features))), dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in dev_features], dtype=torch.long
        )
        all_input_ids_2 = torch.tensor(
            [f.input_ids_2 for f in dev_features], dtype=torch.long
        )
        all_input_mask_2 = torch.tensor(
            [f.input_mask_2 for f in dev_features], dtype=torch.long
        )
        all_segment_ids_2 = torch.tensor(
            [f.segment_ids_2 for f in dev_features], dtype=torch.long
        )
        dev_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_idx,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    else:
        all_input_ids = torch.tensor(
            [f.input_ids for f in dev_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in dev_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in dev_features], dtype=torch.long
        )
        all_guids = [f.guid for f in dev_features]
        all_idx = torch.tensor(list(range(len(dev_features))), dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in dev_features], dtype=torch.long
        )
        dev_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_idx
        )

    # Run prediction for full data
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(
        dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size
    )

    return all_guids, dev_dataloader
