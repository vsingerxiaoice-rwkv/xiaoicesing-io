from copy import deepcopy
from glob import glob
import os
from pathlib import Path
import re
import warnings

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.rank_zero import rank_zero_info

class DiffModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,          
        dirpath,
        filename,
        monitor,
        save_last,
        save_top_k,
        mode,
        max_updates,
        permanent_ckpt_start,
        permanent_ckpt_interval,
        verbose = False,
        save_weights_only = False
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=False
        )
        self.max_updates = max_updates
        self.permanent_ckpt_start = permanent_ckpt_start
        self.permanent_ckpt_interval = permanent_ckpt_interval
        self.last_permanent_step = 0
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the last interrupted training step."""
        if not self._should_skip_saving_checkpoint(trainer) and \
                    trainer.state.stage == RunningStage.TRAINING and \
                    trainer.global_step == self.max_updates:
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer)
                self._save_checkpoint(trainer, filepath)
            self._save_last_checkpoint(trainer, monitor_candidates)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        if trainer.lightning_module.skip_immediate_ckpt_save:
            trainer.lightning_module.skip_immediate_ckpt_save = False
            return
        if not self._should_skip_saving_checkpoint(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)
            
    def state_dict(self):
        ret = super().state_dict()
        ret['last_permanent_step'] = self.last_permanent_step
        return ret

    def load_state_dict(self, state_dict):
        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)

        if self.dirpath == dirpath_from_ckpt:
            self.best_model_score = state_dict["best_model_score"]
            self.kth_best_model_path = state_dict.get("kth_best_model_path", self.kth_best_model_path)
            self.kth_value = state_dict.get("kth_value", self.kth_value)
            self.best_k_models = state_dict.get("best_k_models", self.best_k_models)
            self.last_model_path = state_dict.get("last_model_path", self.last_model_path)
            self.last_permanent_step = state_dict.get("last_permanent_step", self.last_permanent_step)
        else:
            warnings.warn(
                f"The dirpath has changed from {dirpath_from_ckpt!r} to {self.dirpath!r},"
                " therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_permanent_step`,"
                " `last_model_path` and `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded."
            )
        self.best_model_path = state_dict["best_model_path"]
        
    def _monitor_candidates(self, trainer: "pl.Trainer"):
        monitor_candidates = deepcopy(trainer.callback_metrics)
        monitor_candidates["epoch"] = torch.tensor(trainer.current_epoch)
        monitor_candidates["step"] = torch.tensor(trainer.global_step)
        return monitor_candidates
    
    def _save_monitor_checkpoint(self, trainer: "pl.Trainer", monitor_candidates):
        assert self.monitor
        current = monitor_candidates.get(self.monitor)
        if self.check_monitor_top_k(trainer, current):
            assert current is not None
            self._update_best_and_save(current, trainer, monitor_candidates)
        elif self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} was not in top {self.save_top_k}")
            if step >= self.last_permanent_step + self.permanent_ckpt_interval:
                self.last_permanent_step = step
                filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer)
                self._save_checkpoint(trainer, filepath)
                rank_zero_info(f"Epoch {epoch:d}, global step {step:d} is a permanent checkpoint, saved to {filepath}")
    
    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Calls the strategy to remove the checkpoint file."""
        if (self.permanent_ckpt_start or 0) > 0 and (self.permanent_ckpt_interval or 0) > 0:
            search = re.search(r'steps_\d+', Path(filepath).stem)
            if search:
                step = int(search.group(0)[6:])
                if step >= self.permanent_ckpt_start and \
                        (self.last_permanent_step is None or \
                         step >= self.last_permanent_step + self.permanent_ckpt_interval):
                    self.last_permanent_step = step
                    return
        trainer.strategy.remove_checkpoint(filepath)


def get_latest_checkpoint_path(work_dir):
    if not os.path.exists(work_dir):
        return None
    
    last_step = -1
    last_ckpt_name = None

    checkpoints = glob(str(Path(work_dir) / '*.ckpt'))
    for name in checkpoints:
        search = re.search(r'steps_\d+', name)
        if search:
            step = int(search.group(0)[6:])
            if step > last_step:
                last_step = step
                last_ckpt_name = name
                    
    return last_ckpt_name if last_ckpt_name is not None else None


class DiffTQDMProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        for name in ['step', 'batch_size']:
            if name in items:
                items[name] = int(items[name])
        for k, v in items.items():
            if isinstance(v, float):
                if 0.00001 <= v < 10:
                    items[k] = f"{v:.5f}"
        items.pop("v_num", None)
        return items


def get_stategy_obj(strategy):
    if strategy == 'ddp_gloo':
        return DDPStrategy(process_group_backend='gloo')
    else:
        return strategy
