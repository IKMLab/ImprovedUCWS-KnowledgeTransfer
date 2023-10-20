# %%
import argparse
import logging
import os
import subprocess
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from codes import CWSHugTokenizer, InputDataset, OneShotIterator
from codes import set_seed
from codes.segment_classifier import SegmentClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=12000)
parser.add_argument("--data_name", type=str, default='as')
parser.add_argument("--file_name", type=str, default='prediction.txt')
parser.add_argument("--early_stop_threshold", type=int, default=4)
parser.add_argument("--exp", type=str, default='models_42_pipeline_lstm_cls_refactor')
parser.add_argument("--hug_name", type=str, default='bert-base-chinese')
parser.add_argument("--num_labels", type=int, default=2)
parser.add_argument("--out_dir", type=str, default='demo_output_lstm')
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--save_every_steps", type=int, default=4)
parser.add_argument("--lr_rate", type=float, default=5e-5)
parser.add_argument("--max_norm", type=float, default=0.1)

args = parser.parse_args()


def eval(eval_command, out_path):
    out = subprocess.Popen(eval_command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    stdout = stdout.decode("utf-8")

    with open(out_path, 'w') as f_out:
        f_out.write(stdout)

    F_score = 0
    for line in stdout.split('\n')[-15:]:
        if line[:len('=== F MEASURE:')] == '=== F MEASURE:':
            F_score = float(line.split('\t')[-1])

    print(f'{F_score=}')
    return F_score

def set_logger(out_dir):
    log_file = os.path.join(out_dir, 'train.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# Hyper-parameters setting.
data_name = args.data_name # 'cityu'
exp = args.exp # 'models_42_pipeline_lstm_cls_refactor'
hug_name = args.hug_name #'bert-base-chinese'
seed = args.seed
num_labels = args.num_labels
cls_adam_learning_rate = args.lr_rate
cls_train_steps = 1500
save_every_steps = args.save_every_steps
gradient_clip = args.max_norm

# SCRIPT path
out_dir = f'{args.out_dir}/{data_name}_from_exp_seed{seed}'
os.makedirs(out_dir, exist_ok=True)
# os.system(f'rm {out_dir}/events*')

set_logger(out_dir)

DATA_PATH = f'data/{data_name}'
SCRIPT = f'data/score.pl'
TRAINING_WORDS = f'{DATA_PATH}/words.txt'
GOLD_TEST = f'{DATA_PATH}/test_gold.txt'

# cls output and score.
CLS_VALID_OUTPUT = f'{out_dir}/valid_prediction_cls.txt'
CLS_VALID_SCORE = f'{out_dir}/valid_score_cls.txt'
CLS_TEST_OUTPUT = f'{out_dir}/prediction_cls.txt'
CLS_TEST_SCORE = f'{out_dir}/score_cls.txt'

eval_command_valid_cls = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {CLS_VALID_OUTPUT}'

logging.info(f'==== Argument setting ====')
logging.info(f'args: {args}')
logging.info(f'SLM exp: {exp}')
logging.info(f'hug_name: {hug_name}')
logging.info(f'num_labels: {num_labels}')
logging.info(f'cls_adam_learning_rate: {cls_adam_learning_rate}')
logging.info(f'cls_train_steps: {cls_train_steps}')
logging.info(f'save_every_steps: {save_every_steps}')
logging.info(f'gradient_clip: {gradient_clip}')
logging.info(f'{eval_command_valid_cls=}')

# %%
device = torch.device('cuda:0')

set_seed(seed)

vocab_file = 'data/vocab/vocab.txt'
tk = CWSHugTokenizer(
    vocab_file=vocab_file,
    tk_hug_name=hug_name
)
cls_model = SegmentClassifier(
    embedding_size=None,
    vocab_size=None,
    init_embedding=None,
    d_model=768,
    d_ff=None,
    dropout=0.1,
    n_layers=None,
    n_heads=None,
    model_type=hug_name,
    pad_id=tk.pad_id,
    tk_cls=tk,
    encoder=None,
    num_labels=num_labels,
    label_smoothing=0.0,
)

cls_model.to(device)
cls_model.train()


cls_adam_optimizer = optim.Adam(cls_model.parameters(), lr=cls_adam_learning_rate, betas=(0.9, 0.998))
lr_lambda = lambda step: 1 if step < 0.8 * cls_train_steps else 0.1
cls_scheduler = optim.lr_scheduler.LambdaLR(cls_adam_optimizer, lr_lambda=lr_lambda)

logging.info(f'==== Model Info ====')
logging.info(cls_model)
logging.info(f'Tokenizer vocab_size: {len(tk)}')

# %%

training_batch_size, valid_batch_size = args.batch_size, args.batch_size
training_inputs = [f'{exp}/{args.file_name}']
valid_inputs = [f'data/{data_name}/test.txt']

training_dset = InputDataset(
    training_inputs,
    tk,
    is_training=True,
    batch_token_size=training_batch_size
)
training_dldr = DataLoader(
    training_dset,
    num_workers=4,
    batch_size=1,
    shuffle=False,
    collate_fn=InputDataset.single_collate
)
training_iter = OneShotIterator(training_dldr)

# Valid dataloader.
valid_dataset = InputDataset(valid_inputs, tk)
valid_dataloader = DataLoader(
    dataset=valid_dataset,
    shuffle=False,
    batch_size=valid_batch_size,
    num_workers=4,
    collate_fn=InputDataset.padding_collate
)

logging.info('==== Dataset ====')
logging.info(f'{training_batch_size=}')
logging.info(f'{valid_batch_size=}')
logging.info(f'{training_inputs=}')
logging.info(f'{valid_inputs=}')

# %%

logging.info('==== Training Loop ====')
logging.info(f'Training step: {cls_train_steps}')

is_bmes = True if num_labels == 4 else False
writer = SummaryWriter(out_dir)

best_F_score_cls = 0
early_stop_counts, early_stop_threshold = 0, args.early_stop_threshold
start_time = time()

tqdm_bar = tqdm(range(cls_train_steps), dynamic_ncols=True)
for step in tqdm_bar:
    cls_model.train()

    x_batch, seq_len_batch, uchars_batch, segments_batch = next(training_iter)

    batch_labels_cls = cls_model.generate_label(x_batch, segments_batch, tk=tk, is_bmes=is_bmes)

    x_batch = x_batch.to(device)
    loss = cls_model(x=x_batch, labels=batch_labels_cls.to(device))

    loss.backward()

    nn.utils.clip_grad_norm_(cls_model.parameters(), gradient_clip)
    cls_adam_optimizer.step()
    cls_scheduler.step()
    cls_model.zero_grad()
    cls_adam_optimizer.zero_grad()

    tqdm_bar.set_description(f'step: {step}, loss: {loss.item():.4f}')

    writer.add_scalar('loss', loss.item(), step)
    writer.add_scalar('lr', cls_scheduler.get_last_lr()[0], step)

    if (step+1) % save_every_steps == 0:
        cls_model.eval()

        fout_cls = open(CLS_VALID_OUTPUT, 'w')

        with torch.no_grad():
            for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in valid_dataloader:
                x_batch = x_batch.to(device)

                segments_batch_cls = cls_model.generate_segments(x=x_batch, lengths=seq_len_batch)
                for i in restore_orders:
                    uchars, segments_cls = uchars_batch[i], segments_batch_cls[i]
                    fout_cls.write(tk.restore(uchars, segments_cls))

        fout_cls.close()

        F_score_cls = eval(eval_command_valid_cls, CLS_VALID_SCORE)
        writer.add_scalar('F_score_cls', F_score_cls, step)
        logging.info(f'=== step: {step}, F_score_cls: {F_score_cls:.4f} ===')

        if F_score_cls > best_F_score_cls:
            best_F_score_cls = F_score_cls
            logging.info('Saving checkpoint %s...' % out_dir)
            torch.save({
                'global_step': step,
                'best_F_score': best_F_score_cls,
                'model_state_dict': cls_model.state_dict(),
                'adam_optimizer': cls_adam_optimizer.state_dict()
            }, os.path.join(out_dir, 'best-cls_checkpoint'))
            early_stop_counts = 0

            logging.info(f'step: {step}, loss: {loss.item()}, best_F_score: {best_F_score_cls:.4f}')

        elif F_score_cls < best_F_score_cls:
            early_stop_counts += 1

        if early_stop_counts >= early_stop_threshold:
            logging.info(f'Early stop at step {step}.')
            break

        cls_model.train()

writer.close()
print(f'Process time: {time() - start_time }')

# %%
