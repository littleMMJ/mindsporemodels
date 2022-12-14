"""
python lr_schedule.py
"""
import math


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    """linear warmup learning rate"""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    """cosine learning rate"""
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate


def dynamic_lr(config, base_step, rank_size):
    """dynamic learning rate generator"""
    base_lr = config.lr
    total_steps = int(base_step * config.epoch_size)
    warmup_steps = config.warmup_step / rank_size
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(linear_warmup_learning_rate(i + 1, warmup_steps, base_lr, base_lr * config.warmup_ratio))
        else:
            lr.append(a_cosine_learning_rate(i + 1, base_lr, warmup_steps, total_steps - warmup_steps))
    return lr
