from datetime import datetime
from functools import partial
import logging
import os
import pathlib
import shutil
import sys

import matplotlib
matplotlib.use('Agg')

from torch import nn
import torch.utils.data
from torchmetrics import MeanMetric
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_only

from basics.base_model import CategorizedModule
from utils.hparams import hparams
from utils.training_utils import (
    DsModelCheckpoint, DsTQDMProgressBar,
    get_latest_checkpoint_path, get_stategy
)
from utils.phoneme_utils import locate_dictionary

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class BaseTask(pl.LightningModule):
    '''
        Base class for training tasks.
        1. *load_ckpt*:
            load checkpoint;
        2. *training_step*:
            record and log the loss;
        3. *optimizer_step*:
            run backwards step;
        4. *start*:
            load training configs, backup code, log to tensorboard, start training;
        5. *configure_ddp* and *init_ddp_connection*:
            start parallel training.

        Subclasses should define:
        1. *build_model*, *build_optimizer*, *build_scheduler*:
            how to build the model, the optimizer and the training scheduler;
        2. *_training_step*:
            one training step of the model;
        3. *on_validation_end* and *_on_validation_end*:
            postprocess the validation output.
    '''

    def __init__(self, *args, **kwargs):
        # dataset configs
        super(BaseTask, self).__init__(*args, **kwargs)
        self.loaded_optimizer_states_dict = {}
        self.example_input_array = None

        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_eval_tokens = hparams['max_eval_tokens']
        if self.max_eval_tokens == -1:
            hparams['max_eval_tokens'] = self.max_eval_tokens = self.max_tokens
        self.max_eval_sentences = hparams['max_eval_sentences']
        if self.max_eval_sentences == -1:
            hparams['max_eval_sentences'] = self.max_eval_sentences = self.max_sentences

        self.training_sampler = None
        self.model = None
        
        self.valid_metrics = {
            'total_loss': MeanMetric()
        }

    ###########
    # Training, validation and testing
    ###########
    
    def build_model(self):
        raise NotImplementedError

    def on_train_epoch_start(self):
        if self.training_sampler is not None:
            self.training_sampler.set_epoch(self.current_epoch)

    def _training_step(self, sample, batch_idx, optimizer_idx):
        """

        :param sample:
        :param batch_idx:
        :return: total loss: torch.Tensor, loss_log: dict, other_log: dict
        """
        raise NotImplementedError

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        total_loss, log_outputs = self._training_step(sample, batch_idx, optimizer_idx)
        
        # logs to progress bar
        self.log_dict(log_outputs, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log('lr', self.lr_schedulers().get_lr()[0], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        # logs to tensorboard
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        self.log_dict(tb_log, logger=True, on_step=True, on_epoch=False)
        
        return total_loss
    
    # def on_before_optimizer_step(self, *args, **kwargs):
    #     self.log_dict(grad_norm(self, norm_type=2))
    
    def _on_validation_start(self):
        pass
    
    def on_validation_start(self):
        self._on_validation_start()
        for metric in self.valid_metrics.values():
            metric.to(self.device)
            metric.reset()

    def _validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: loss_log: dict, weight: int
        """
        raise NotImplementedError

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        """
        with torch.autocast('cuda' if next(self.model.parameters()).is_cuda else 'cpu', dtype=torch.float32):
            outputs, weight = self._validation_step(sample, batch_idx)
        for k, v in outputs.items():
            if isinstance(self.valid_metrics[k], MeanMetric):
                self.valid_metrics[k].update(v, weight=weight)

    def on_validation_epoch_end(self):
        metric_vals = {k: v.compute() for k, v in self.valid_metrics.items()}
        self.log('val_loss', metric_vals['total_loss'], on_epoch=True, prog_bar=True, logger=False)
        self.log_dict({f'val/{k}': v for k, v in metric_vals.items()}, on_epoch=True, logger=True)
        for metric in self.valid_metrics.values():
            metric.reset()

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def configure_optimizers(self):
        optm = self.build_optimizer(self.model)
        scheduler = self.build_scheduler(optm)
        if scheduler is None:
            return optm
        return {
            "optimizer": optm,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def on_test_start(self):
        self.on_validation_start()

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def on_test_end(self):
        return self.on_validation_end()

    ###########
    # Running configuration
    ###########

    @classmethod
    def start(cls):
        pl.seed_everything(hparams['seed'], workers=True)
        task = cls()
        work_dir = pathlib.Path(hparams['work_dir'])
        trainer = pl.Trainer(
            accelerator=hparams['pl_trainer_accelerator'],
            devices=hparams['pl_trainer_devices'],
            num_nodes=hparams['pl_trainer_num_nodes'],
            strategy=get_stategy(
                accelerator=hparams['pl_trainer_accelerator'],
                devices=hparams['pl_trainer_devices'],
                num_nodes=hparams['pl_trainer_num_nodes'],
                strategy=hparams['pl_trainer_strategy'],
                backend=hparams['ddp_backend']
            ),
            precision=hparams['pl_trainer_precision'],
            callbacks=[
                DsModelCheckpoint(
                    dirpath=work_dir,
                    filename='model_ckpt_steps_{step}',
                    auto_insert_metric_name=False,
                    monitor='step',
                    mode='max',
                    save_last=False,
                    every_n_train_steps=hparams['val_check_interval'],
                    save_top_k=hparams['num_ckpt_keep'],
                    permanent_ckpt_start=hparams['permanent_ckpt_start'],
                    permanent_ckpt_interval=hparams['permanent_ckpt_interval'],
                    verbose=True
                ),
                LearningRateMonitor(logging_interval='step'),
                DsTQDMProgressBar(),
            ],
            logger=TensorBoardLogger(
                save_dir=str(work_dir),
                name='lightning_logs',
                version='lastest'
            ),
            gradient_clip_val=hparams['clip_grad_norm'],
            val_check_interval=hparams['val_check_interval'] * hparams['accumulate_grad_batches'], # so this is global_steps
            check_val_every_n_epoch=None,
            log_every_n_steps=hparams['log_interval'],
            max_steps=hparams['max_updates'],
            use_distributed_sampler=False,
            num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams['validate'] else 10000,
            accumulate_grad_batches=hparams['accumulate_grad_batches']
        )
        if not hparams['infer']:  # train
            @rank_zero_only
            def train_payload_copy():
                # copy_code = input(f'{hparams["save_codes"]} code backup? y/n: ') == 'y'
                copy_code = True  # backup code every time
                if copy_code:
                    code_dir = work_dir / 'codes' / datetime.now().strftime('%Y%m%d%H%M%S')
                    code_dir.mkdir(exist_ok=True, parents=True)
                    for c in hparams['save_codes']:
                        shutil.copytree(c, code_dir, dirs_exist_ok=True)
                    print(f'| Copied codes to {code_dir}.')
                # Copy spk_map.json and dictionary.txt to work dir
                binary_dir = pathlib.Path(hparams['binary_data_dir'])
                spk_map = work_dir / 'spk_map.json'
                spk_map_src = binary_dir / 'spk_map.json'
                if not spk_map.exists() and spk_map_src.exists():
                    shutil.copy(spk_map_src, spk_map)
                    print(f'| Copied spk map to {spk_map}.')
                dictionary = work_dir / 'dictionary.txt'
                dict_src = binary_dir / 'dictionary.txt'
                if not dictionary.exists():
                    if dict_src.exists():
                        shutil.copy(dict_src, dictionary)
                    else:
                        shutil.copy(locate_dictionary(), dictionary)
                    print(f'| Copied dictionary to {dictionary}.')
            train_payload_copy()
            trainer.fit(task, ckpt_path=get_latest_checkpoint_path(work_dir))
        else:
            trainer.test(task)

    def on_save_checkpoint(self, checkpoint):
        if isinstance(self.model, CategorizedModule):
            checkpoint['category'] = self.model.category
        checkpoint['trainer_stage'] = self.trainer.state.stage.value
    
    def on_load_checkpoint(self, checkpoint):
        from lightning.pytorch.trainer.states import RunningStage
        if checkpoint.get('trainer_stage', '') == RunningStage.VALIDATING.value:
            self.skip_immediate_validation = True
