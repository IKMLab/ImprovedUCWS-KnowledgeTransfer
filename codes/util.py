import logging
import pickle
import subprocess

import numpy as np
import torch
import torch.nn as nn


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def set_logger(log_file):
    r"""Write logs to checkpoint and console."""

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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_module(module):
    """Initialize the weights from huggingface BERT"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def eval(eval_command, out_path, is_pred=False, **kwargs):
    logging.info(eval_command)
    out = subprocess.Popen(eval_command.split(
        ' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    stdout = stdout.decode("utf-8")

    with open(out_path, 'w') as f_out:
        f_out.write(stdout)

    tail_info = stdout.split('\n')[-15:]
    log_info = f'\nTest results:\n%s' % '\n'.join(
        tail_info) if is_pred else f'\nValidation results:\n%s' % '\n'.join(tail_info)
    logging.info(log_info)

    F_score = 0
    for line in tail_info:
        if line[:len('=== F MEASURE:')] == '=== F MEASURE:':
            F_score = float(line.split('\t')[-1])

    print(f'{F_score=}')
    return F_score
