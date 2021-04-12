import os
import wandb
import torch
import warnings
from time import sleep
from pathlib import Path
from poutyne.framework import Callback, ModelCheckpoint


class WandbCallback(Callback):

    def __init__(self,
                 project=None,
                 name=None,
                 config=None,
                 entity=None,
                 prefix='',
                 period=1):
        super().__init__()
        if wandb.run is None:
            wandb.init(project=project, name=name, entity=entity)
        if config:
            wandb.config.update(config)

        if len(prefix) > 0 and not (prefix.endswith('_') or prefix.endswith(' ')):
            prefix = prefix + '_'

        self.prefix = prefix
        self.period = period
        self.steps_elapsed = 0

    def on_epoch_end(self, epoch, logs):

        logs.update(epoch=epoch)

        logs = {self.prefix + key: value for key, value in logs.items()}
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

            logs = {self.prefix + key: value for key, value in logs.items()}
            wandb.log(logs, sync=False, commit=True)


class DummySyncCallback(Callback):
    def __init__(self,
                 save_dir,
                 output_name='output_{step}_{rank_id}',
                 input_name='input_{step}',
                 rank_id=0,
                 period=10,
                 serialize_fn=None,
                 deserialize_fn=None,
                 max_tries=100,
                 wait_time=0.1,
                 ):
        super().__init__()
        self.save_dir = save_dir
        self.rank_id = rank_id
        self.output_name = output_name
        self.input_name = input_name
        self.period = period
        self.serialize_fn = serialize_fn if serialize_fn else self._default_serialize_fn
        self.deserialize_fn = deserialize_fn if deserialize_fn else self._default_deserialize_fn
        self.steps_elapsed = 0
        self.max_tries = max_tries
        self.wait_time = wait_time
        self.is_synced = False

    @staticmethod
    def _default_serialize_fn(state_dict):
        return state_dict

    @staticmethod
    def _default_deserialize_fn(state_dict):
        return state_dict

    @staticmethod
    def save_weights(state_dict, fd):
        torch.save(state_dict, fd)

    @staticmethod
    def load_weights(fd):
        return torch.load(fd, map_location='cpu')

    def send_weights(self, state_dict):
        filename = os.path.join(
            self.save_dir,
            self.output_name.format(step=self.steps_elapsed, rank_id=self.rank_id)
        )
        self.save_weights(state_dict, filename)

    def recv_weights(self):
        filename = os.path.join(
            self.save_dir,
            self.input_name.format(step=self.steps_elapsed)
        )

        state_dict = None
        for _ in range(self.max_tries):
            try:
                state_dict = self.load_weights(filename)
                break
            except:
                sleep(self.wait_time)
        return state_dict

    def _synchronize(self):
        weights = self.serialize_fn(self.model.network.state_dict())
        self.send_weights(weights)

        weights = self.recv_weights()
        weights = self.deserialize_fn(weights)
        self.model.set_weights(weights)

    def on_train_batch_end(self, batch_number: int, logs):
        self.steps_elapsed += 1
        if self.steps_elapsed % self.period == 0:
            self._synchronize()
            self.is_synced = True
        else:
            self.is_synced = False

    def on_test_batch_begin(self, batch_number: int, logs):
        if not self.is_synced:
            self._synchronize()
            self.is_synced = True


class MasterDummySyncCallback(DummySyncCallback):
    def __init__(self,
                 save_dir,
                 output_name='output_{step}_{rank_id}',
                 input_name='input_{step}',
                 rank_id=0,
                 period=10,
                 serialize_fn=None,
                 deserialize_fn=None,
                 n_workers=1,
                 max_tries=100,
                 wait_time=0.1
                 ):
        super().__init__(save_dir,
                         output_name=output_name,
                         input_name=input_name,
                         rank_id=rank_id,
                         period=period,
                         serialize_fn=serialize_fn,
                         deserialize_fn=deserialize_fn,
                         wait_time=wait_time,
                         max_tries=max_tries)
        self.n_workers = n_workers

    def get_weight_dumps(self):
        output_glob = self.output_name.format(step=self.steps_elapsed, rank_id='*')
        return list(Path(self.save_dir).glob(output_glob))

    def aggregate_weights(self, weights_dump):
        weights_dump = [self.deserialize_fn(w) for w in weights_dump]
        aggregated_state_dict = dict()

        for name in weights_dump[0].keys():
            if name.endswith('.weight') and name.endswith('.bias'):
                weights = [state_dict[name] for state_dict in weights_dump]
                aggregated_state_dict[name] = torch.stack(weights, dim=0).mean(dim=0)
            else:
                aggregated_state_dict[name] = weights_dump[0][name]
        return aggregated_state_dict

    def aggregate_dumps(self):
        while len(self.get_weight_dumps()) < self.n_workers:
            # waiting other workers
            sleep(self.wait_time)

        weights_dump = None
        for _ in range(self.max_tries):
            try:
                dumps = self.get_weight_dumps()
                weights_dump = [self.load_weights(f) for f in dumps]
                break
            except:
                sleep(self.wait_time)

        return self.aggregate_weights(weights_dump)

    def _clean_up(self, filename_glob):
        files = list(Path(self.save_dir).glob(filename_glob))
        for f in files:
            os.remove(str(f))
        pass

    def send_weights(self, state_dict):
        filename = os.path.join(
            self.save_dir,
            self.output_name.format(step=self.steps_elapsed, rank_id=self.rank_id)
        )
        self.save_weights(state_dict, filename)

        state_dict = self.aggregate_dumps()
        self._clean_up(self.input_name.format(step='*'))

        filename = os.path.join(
            self.save_dir,
            self.input_name.format(step=self.steps_elapsed)
        )
        state_dict = self.serialize_fn(state_dict)
        self.save_weights(state_dict, filename)
        self._clean_up(self.output_name.format(step='*', rank_id='*'))
