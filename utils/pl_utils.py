from copy import deepcopy
import os
import re

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

class DiffModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _monitor_candidates(self, trainer: "pl.Trainer"):
        monitor_candidates = deepcopy(trainer.callback_metrics)
        monitor_candidates["epoch"] = torch.tensor(trainer.current_epoch)
        monitor_candidates["step"] = torch.tensor(trainer.global_step)
        return monitor_candidates

    def _should_save_on_train_epoch_end(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import RunningStage
        return trainer.state.stage == RunningStage.TRAINING and super()._should_save_on_train_epoch_end(trainer)

    # @classmethod
    # def _format_checkpoint_name(cls, filename, metrics, prefix = "", auto_insert_metric_name = True):
    #     # metrics = {k: v + 1 if k == 'step' or k == 'epoch' else v for k, v in metrics.items()}
    #     return super()._format_checkpoint_name(filename, metrics, prefix, auto_insert_metric_name)

def get_latest_checkpoint_path(work_dir):
    if not os.path.exists(work_dir):
        return None
    
    last_steps = -1
    last_ckpt_name = None

    checkpoints = os.listdir(work_dir)
    for name in checkpoints:
        if '.ckpt' in name and not name.endswith('part'):
            if 'steps_' in name:
                steps = name.split('steps_')[1]
                steps = int(re.sub('[^0-9]', '', steps))

                if steps > last_steps:
                    last_steps = steps
                    last_ckpt_name = name
                    
    return os.path.join(work_dir, last_ckpt_name) if last_ckpt_name is not None else None

def get_stategy_obj(strategy):
    if strategy == 'ddp_gloo':
        return DDPStrategy(process_group_backend='gloo')
    else:
        return 'auto'
