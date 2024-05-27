def get_new_lr(current_epoch: int, total_epoch: int, decay_epoch: int, initial_lr: float, target_lr: float) -> float:
    if current_epoch < decay_epoch:
        return initial_lr
    else:
        k = (target_lr - initial_lr) / (total_epoch - decay_epoch)
        return initial_lr + (current_epoch - decay_epoch) * k