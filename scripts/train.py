import os
import importlib
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1' # Prevent unacceptable slowdowns when using 16 precision

from utils.hparams import set_hparams, hparams

set_hparams()
hparams['disable_sample_tqdm'] = True
if hparams['ddp_backend'] == 'nccl_no_p2p':
    os.environ['NCCL_P2P_DISABLE'] = '1'

def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()


if __name__ == '__main__':
    run_task()
