import wandb
import torch
import warnings
from poutyne.framework import Callback, ModelCheckpoint


class WandbCallback(Callback):

    def __init__(self,
                 project_name=None,
                 name=None,
                 config=None,
                 period=1):
        super().__init__()
        if wandb.run is None:
            wandb.init(project=project_name, name=name)
        if config:
            wandb.config.update(config)

        self.period = period
        self.steps_elapsed = 0

    def on_epoch_end(self, epoch, logs):

        logs.update(epoch=epoch)
        wandb.log(logs, sync=False, commit=True)

    def on_train_batch_end(self, batch, logs):
        self.steps_elapsed += 1
        if self.steps_elapsed % self.period == 0:

            if hasattr(self.model.optimizer, 'get_lr'):
                learning_rates = [self.model.optimizer.get_lr()[0]]
            else:
                learning_rates = (param_group['lr'] for param_group in self.model.optimizer.param_groups)
            for group_idx, lr in enumerate(learning_rates):
                logs['learning_rate_group_' + str(group_idx)] = lr
            logs['step'] = self.steps_elapsed
            wandb.log(logs, sync=False, commit=True)
