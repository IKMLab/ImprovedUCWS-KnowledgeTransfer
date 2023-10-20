r"""Second stage for perturbed tokenization result.

Example:

    python -m codes.perturb_second_stage \
        --data_name as \
        --data_type test \
        --bound 11.0 \
        --out_dir exp/perturbed/second_stage

"""
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
from codes.segment_classifier import SegmentClassifier
from codes import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='as')
parser.add_argument("--data_type", type=str, default='train')
parser.add_argument("--bound", type=float, default=11.0)
parser.add_argument("--num_labels", type=int, default=2)
parser.add_argument("--exp", type=str, default='models_42_pipeline_lstm_cls_refactor')
parser.add_argument("--out_dir", type=str, default='demo_output_perturb')
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()


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


# Hyper-parameters setting.
data_name = args.data_name # 'cityu'
data_type = args.data_type
exp = args.exp
num_labels = args.num_labels
seed = args.seed
hug_name = 'bert-base-chinese'
cls_adam_learning_rate = 5e-5
cls_train_steps = 1500
save_every_steps = 400
gradient_clip = 0.1

if args.bound is not None:
    exp = f'log/perturbed_masking/bert-base-chinese/bound{args.bound}/{data_name}_{data_type}.txt'


out_dir = f'{args.out_dir}/{args.data_name}'
os.makedirs(out_dir, exist_ok=True)

set_logger(out_dir)

DATA_PATH = f'data/{data_name}'
SCRIPT = f'data/score.pl'
TRAINING_WORDS = f'{DATA_PATH}/words.txt'
GOLD_TEST = f'{DATA_PATH}/test_gold.txt'

# Cls valid.
CLS_VALID_OUTPUT = f'{out_dir}/valid_prediction_cls.txt'
CLS_VALID_SCORE = f'{out_dir}/valid_score_cls.txt'
# Cls prediction.
CLS_TEST_OUTPUT = f'{out_dir}/prediction_cls.txt'
CLS_TEST_SCORE = f'{out_dir}/score_cls.txt'

eval_command = f'perl {SCRIPT} {TRAINING_WORDS} {GOLD_TEST} {CLS_VALID_OUTPUT}'

logging.info(f'==== Argument setting ====')
logging.info(f'args: {args}')
logging.info(f' exp: {exp}')
logging.info(f'data_type: {data_type}')
logging.info(f'hug_name: {hug_name}')
logging.info(f'cls_adam_learning_rate: {cls_adam_learning_rate}')
logging.info(f'cls_train_steps: {cls_train_steps}')
logging.info(f'save_every_steps: {save_every_steps}')
logging.info(f'gradient_clip: {gradient_clip}')
logging.info(f'{eval_command=}')

# %%
device = torch.device('cuda:0')
set_seed(seed)

tk = CWSHugTokenizer(
    vocab_file=None,
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
    num_labels=2,
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
unsupervised_batch_size = 12000
valid_batch_size = 12000
unsegmented = [f'{exp}/unsupervised-{data_name}-4/prediction.txt']
valid_inputs = [f'data/{data_name}/test.txt']

if args.bound is not None:
    unsegmented = [f'log/perturbed_masking/bert-base-chinese/bound{args.bound}/{data_name}_{data_type}.txt']

unsupervsied_dataset = InputDataset(
    unsegmented,
    tk,
    is_training=True,
    batch_token_size=unsupervised_batch_size
)
unsupervised_dataloader = DataLoader(
    unsupervsied_dataset,
    num_workers=4,
    batch_size=1,
    shuffle=False,
    collate_fn=InputDataset.single_collate
)
unsupervised_data_iterator = OneShotIterator(unsupervised_dataloader)

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
logging.info(f'{unsupervised_batch_size=}')
logging.info(f'{valid_batch_size=}')
logging.info(f'{unsegmented=}')
logging.info(f'{valid_inputs=}')

# %%

logging.info('==== Training Loop ====')
logging.info(f'Training step: {cls_train_steps}')

writer = SummaryWriter(out_dir)

best_F_score_cls = 0
early_stop_counts, early_stop_threshold = 0, 4
start_time = time()

tqdm_bar = tqdm(range(cls_train_steps))
for step in tqdm_bar:
    cls_model.train()

    x_batch, seq_len_batch, uchars_batch, segments_batch = next(unsupervised_data_iterator)
    batch_labels_cls = []
    for input_ids, segment in zip(x_batch, segments_batch):
        e_idx = (input_ids == tk.eos_id).nonzero(as_tuple=True)[0]
        label = torch.zeros(input_ids.size(0))
        label[e_idx+1:].fill_(-100)
        idx = [sum(segment[:i])-1+1 for i in range(1, len(segment)+1)]
        label[idx] = 1
        batch_labels_cls.append(label)

    x_batch = x_batch.to(device)
    labels_cls = torch.stack(batch_labels_cls, 0).long().to(device)
    loss = cls_model(x=x_batch, labels=labels_cls)

    loss.backward()

    nn.utils.clip_grad_norm_(cls_model.parameters(), gradient_clip)
    cls_adam_optimizer.step()
    cls_scheduler.step()
    cls_model.zero_grad()
    cls_adam_optimizer.zero_grad()

    tqdm_bar.set_description(f'step: {step}, loss: {loss.item():.4f}')

    writer.add_scalar('loss', loss.item(), step)
    writer.add_scalar('lr', cls_scheduler.get_last_lr()[0], step)

    if (step+1) % save_every_steps:
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

        F_score_cls = eval(eval_command, CLS_VALID_SCORE)
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
