import torch.optim as optim
from transformers import (get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup)


def get_optimizer_and_scheduler(
    model, lr_rate: float, optimizer_type='Adam',
    lr_lambda=None, scheduler_type=None, scheduler_args=None,
):

    # Optimizer part.
    if optimizer_type == 'AdamW':
        param_optimizer = model.named_parameters()
        weight_decay = 0.01
        no_decay = ['bias', 'LayerNorm.weight', 'ln_1.weight', 'ln_2.weight']

        optimizer_grouped_parameters = [
            {
                'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr_rate)

    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr_rate, betas=(0.9, 0.998))

    # Scheduler part.
    if lr_lambda is not None:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = {
            'constant': get_constant_schedule_with_warmup,
            'cosine': get_cosine_schedule_with_warmup,
            'linear': get_linear_schedule_with_warmup,
        }[scheduler_type](optimizer=optimizer, **scheduler_args)

    return optimizer, scheduler
