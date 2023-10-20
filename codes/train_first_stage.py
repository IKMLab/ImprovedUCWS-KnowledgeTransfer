r"""The first stage of the framework.
The segment model is trained on the raw corpus without any human annotation.
"""
import argparse
import logging
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from codes import (SegmentalLM, SLMConfig, CWSHugTokenizer, InputDataset,
    OneShotIterator, set_logger, set_seed, eval, get_optimizer_and_scheduler)


def get_args():
    parser = argparse.ArgumentParser()

    # Mode.
    parser.add_argument("--do_train", action='store_true', help="Whether to run unsupervised training.")
    parser.add_argument("--do_valid", action='store_true', help="Whether to do validation during training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation of test set.")
    parser.add_argument("--do_eval_train", action='store_true', help="Whether to evaluate the training set.")

    # General setting.
    parser.add_argument("--unsegmented", type=str, nargs="+", help="Path of unsegmented input file.")

    parser.add_argument("--max_seq_length", type=int, default=32, help="The maximum sequence length.")

    parser.add_argument("--data_name", type=str, required=True, help="CWS dataset.")
    parser.add_argument("--config_file", type=str, required=True, help="Path of the SLM configuration file.")
    parser.add_argument("--save_dir", type=str, help="Path of the saving checkpoint.")

    # Training setting.
    parser.add_argument("--gradient_clip", type=float, default=0.1)
    parser.add_argument("--sgd_lr_rate", type=float, default=16.0)
    parser.add_argument("--adam_lr_rate", type=float, default=0.005)

    parser.add_argument("--train_batch_size", type=int, default=6000)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    parser.add_argument("--train_steps", type=int, default=6000)
    parser.add_argument("--warm_up_steps", type=int, default=800)
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--save_every_steps", type=int, default=400)

    # Token setting.
    parser.add_argument("--segment_token", type=str, default='  ', help="Segment token.")
    parser.add_argument("--english_token", type=str, default='<ENG>', help="Token for English characters.")
    parser.add_argument("--number_token", type=str, default='<NUM>', help="Token for numbers.")
    parser.add_argument("--punctuation_token", type=str, default='<PUNC>', help="Token for punctuations.")
    parser.add_argument("--bos_token", type=str, default='<BOS>', help="Token for begin of sentence.")
    parser.add_argument("--eos_token", type=str, default='</s>', help="Token for end of sentence.")

    parser.add_argument("--hug_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42) # 42, 87, 5253

    return parser.parse_args()


def main(args):

    logging.info(f'args: {args}')

    set_seed(args.seed)

    device = torch.device('cuda:0')

    config = SLMConfig.from_json_file(args.config_file)
    logging.info(f'Config Info:\n{config.to_json_string()}')

    init_embedding = None
    if config.embedding_path:
        logging.info(f'Loading word embedding from {config.embedding_path}.')
        init_embedding = np.load(config.embedding_path)

    tokenizer = CWSHugTokenizer(
        vocab_file=config.vocab_file,
        tk_hug_name=args.hug_name,
        max_seq_length=args.max_seq_length,
        segment_token=args.segment_token,
        english_token=args.english_token,
        number_token=args.number_token,
        punctuation_token=args.punctuation_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
    )

    config.vocab_size = len(tokenizer)

    slm = SegmentalLM(
        config=config,
        init_embedding=init_embedding,
        hug_name=args.hug_name
    )
    logging.info('Model Info:\n%s' % slm)
    slm.to(device)

    # Path setting.
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
    DATA_PATH = f'{PROJECT_ROOT_PATH}/data/{args.data_name}'

    # score script, words and gold.
    SCRIPT = f'{PROJECT_ROOT_PATH}/data/score.pl'
    TRAINING_WORDS = f'{DATA_PATH}/words.txt'
    GOLD_TEST = f'{DATA_PATH}/test_gold.txt'

    MODEL_DIR = args.save_dir

    # Score and predcition output file.
    # Validation output file.
    VALID_OUTPUT = f'{MODEL_DIR}/valid_prediction.txt'
    VALID_SCORE = f'{MODEL_DIR}/valid_score.txt'

    # Evaluation output file.
    TEST_OUTPUT = f'{MODEL_DIR}/prediction.txt'
    TEST_SCORE = f'{MODEL_DIR}/score.txt'

    TRAIN_DATA = f'{DATA_PATH}/unsegmented.txt'
    TEST_DATA = f'{DATA_PATH}/test.txt'

    train_inputs = [TRAIN_DATA, TEST_DATA]
    valid_inputs = [TEST_DATA]
    eval_inputs = [TEST_DATA]

    if args.do_train:
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=slm,
            lr_rate=args.adam_lr_rate,
            optimizer_type='Adam',
            lr_lambda=lambda step: 1 if step < 0.8 * args.train_steps else 0.1,
        )

        logging.info('Prepare unsupervised dataset.')
        train_dset = InputDataset(
            train_inputs,
            tokenizer,
            is_training=True,
            batch_token_size=args.train_batch_size
        )
        train_dldr = DataLoader(
            train_dset,
            num_workers=4,
            batch_size=1,
            shuffle=False,
            collate_fn=InputDataset.single_collate
        )
        train_iter = OneShotIterator(train_dldr)

        if args.do_valid:
            logging.info('Prepare validation dataloader')
            valid_dset = InputDataset(valid_inputs, tokenizer)
            valid_dldr = DataLoader(
                dataset=valid_dset,
                shuffle=False,
                batch_size=args.valid_batch_size,
                num_workers=4,
                collate_fn=InputDataset.padding_collate
            )

        logging.info('Ramdomly Initializing SLM parameters...')
        global_step = 0

        # Tensrboard writer.
        writer = SummaryWriter(args.save_dir)

        basic_arguments = {
            'args': args,
            'device': device,
            'eval_command': f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {VALID_OUTPUT}',
            'tokenizer': tokenizer,
            'config': config,
            'VALID_OUTPUT': VALID_OUTPUT,
            'VALID_SCORE': VALID_SCORE,
            'train_iter': train_iter,
            'valid_dldr': valid_dldr,
            'writer': writer
        }

        # Train the SLM firstly.
        train_slm(
            start_step=global_step,
            end_step=args.train_steps,
            slm=slm,
            optimizer=optimizer,
            scheduler=scheduler,
            **basic_arguments,
        )

        writer.close()

    if args.do_eval:
        predict_method(args, device, tokenizer, slm, eval_inputs,
            SCRIPT, TRAINING_WORDS, GOLD_TEST, TEST_OUTPUT, TEST_SCORE)

    if args.do_eval_train:
        GOLD_TEST = f'{DATA_PATH}/segmented.txt'
        TEST_OUTPUT = f'{MODEL_DIR}/prediction_train.txt'
        TEST_SCORE = f'{MODEL_DIR}/score_train.txt'
        predict_method(args, device, tokenizer, slm, eval_inputs,
            SCRIPT, TRAINING_WORDS, GOLD_TEST, TEST_OUTPUT, TEST_SCORE)


def train_slm(
    start_step, end_step,
    args, device, eval_command, tokenizer, config, slm,
    VALID_OUTPUT, VALID_SCORE, optimizer, scheduler, train_iter, valid_dldr,
    writer,
    **kwargs
):
    logging.info('==== SLM training loop. ====')
    config.to_json_file(os.path.join(args.save_dir, 'config.json'))

    best_F_score =  0.0

    logs = []
    for step in tqdm(range(start_step, end_step), dynamic_ncols=True):
        slm.train()

        log = {}

        x_batch, seq_len_batch, uchars_batch, segments_batch = next(train_iter)
        x_batch = x_batch.to(device)
        loss = slm(x_batch, seq_len_batch, mode='unsupervised')
        log['unsupervised_loss'] = loss.item()

        logs.append(log)

        loss.backward()
        nn.utils.clip_grad_norm_(slm.parameters(), args.gradient_clip)

        if step > args.warm_up_steps:
            optimizer.step()
        else:
            # Do manually SGD.
            for p in slm.parameters():
                if p.grad is not None:
                    p.data.add_(p.grad.data, alpha=-args.sgd_lr_rate)

        scheduler.step()
        optimizer.zero_grad()
        slm.zero_grad()

        if (step+1) % args.log_every_steps == 0:
            logging.info("global_step = %s" % step)
            if len(logs) > 0:
                for key in logs[0]:
                    logging.info("%s = %f" % (key, sum([log[key] for log in logs])/len(logs)))
                    writer.add_scalar(key, sum([log[key] for log in logs])/len(logs), step)
            else:
                logging.info("Currently no metrics available")
            logs = []

        if ((step+1) % args.save_every_steps == 0) or (step == args.train_steps - 1):
            logging.info('Saving checkpoint %s...' % args.save_dir)
            torch.save({
                'global_step': step,
                'best_F_score': best_F_score,
                'model_state_dict': slm.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(args.save_dir, 'checkpoint'))

            if args.do_valid:
                slm.eval()
                fout_slm = open(VALID_OUTPUT, 'w')
                with torch.no_grad():
                    for x_batch_val, seq_len_batch, uchars_batch, segments_batch, restore_orders in valid_dldr:
                        torch.cuda.empty_cache()
                        x_batch_val = x_batch_val.to(device)
                        segments_batch_slm = slm(x_batch_val, seq_len_batch, mode='decode')
                        for i in restore_orders:
                            uchars, segments_slm = uchars_batch[i], segments_batch_slm[i]
                            fout_slm.write(tokenizer.restore(uchars, segments_slm))

                fout_slm.close()

                F_score = eval(eval_command, VALID_SCORE)
                writer.add_scalar('F_score', F_score, step)

                if (step + 1) % 1000 == 0:
                    name = VALID_OUTPUT.replace('.txt', f'{step}.txt')
                    os.system(f'cp {VALID_OUTPUT} {name}')
                    logging.info(f'Save the segmentations result at step {step+1}')

            if (not args.do_valid) or (F_score > best_F_score):
                best_F_score = F_score
                logging.info('Overwriting best checkpoint....')
                os.system('cp %s %s' % (os.path.join(args.save_dir, 'checkpoint'),
                                            os.path.join(args.save_dir, 'best-checkpoint')))


def predict_method(
    args, device, tokenizer, slm, eval_inputs,
    SCRIPT, TRAINING_WORDS, GOLD_TEST, TEST_OUTPUT, TEST_SCORE,
    **kwargs
):
    logging.info('Prepare prediction dataloader')
    logging.info('===== Start to SLM predict =====')

    predict_dataset = InputDataset(eval_inputs, tokenizer)
    predict_dataloader = DataLoader(
        dataset=predict_dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=4,
        collate_fn=InputDataset.padding_collate
    )

    # Load best checkpoint model.
    logging.info(f'Loading checkpoint {args.save_dir}.')
    checkpoint = torch.load(os.path.join(args.save_dir, 'best-checkpoint'))
    step = checkpoint['global_step']
    slm.load_state_dict(checkpoint['model_state_dict'])
    slm.eval()

    logging.info(f'Global step of best-checkpoint: {step}')

    fout_slm = open(TEST_OUTPUT, 'w')

    with torch.no_grad():
        for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in predict_dataloader:
            x_batch = x_batch.to(device)
            segments_batch_slm = slm(x_batch, seq_len_batch, mode='decode')
            for i in restore_orders:
                uchars, segments_slm = uchars_batch[i], segments_batch_slm[i]
                fout_slm.write(tokenizer.restore(uchars, segments_slm))

    fout_slm.close()

    eval_command_slm = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {TEST_OUTPUT}'
    F_score = eval(eval_command_slm, TEST_SCORE, True)


if __name__ == "__main__":

    args = get_args()

    # Create the save directory.
    os.makedirs(args.save_dir, exist_ok=True)

    # Set the log file.
    set_logger(os.path.join(args.save_dir, 'train.log'))

    start_time = time()

    main(args)

    logging.info(f'Process time: {time() - start_time }')


