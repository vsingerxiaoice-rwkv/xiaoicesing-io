import torch
import torch.nn as nn

cls_map = {'fs2': 'modules.shallow.fast_speech2_decoder.fs2_decode'}


def build_object_from_class_name(cls_str, parent_cls, strict, *args, **kwargs):
    import importlib

    pkg = ".".join(cls_str.split(".")[:-1])
    cls_name = cls_str.split(".")[-1]
    cls_type = getattr(importlib.import_module(pkg), cls_name)
    if parent_cls is not None:
        assert issubclass(cls_type, parent_cls), f'| {cls_type} is not subclass of {parent_cls}.'
    if strict:
        return cls_type(*args, **kwargs)
    return cls_type(*args, **filter_kwargs(kwargs, cls_type))


def filter_kwargs(dict_to_filter, kwarg_obj):
    import inspect

    sig = inspect.signature(kwarg_obj)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys if
                     filter_key in dict_to_filter}
    return filtered_dict


class shallow_adapt(nn.Module):
    def __init__(self, parame, out_dims):
        super().__init__()
        self.parame = parame

        decodeparame=parame['shallow_diffusion_args']['aux_decoder_args']
        decodeparame[ 'encoder_hidden'] = parame['hidden_size']
        decodeparame['out_dims'] = out_dims
        decodeparame['parame'] = parame

        self.model = build_object_from_class_name(cls_map[parame['shallow_diffusion_args']['aux_decoder_arch']],
                                                  nn.Module,
                                                  parame['shallow_diffusion_args']['aux_decode_strict_hparams'],
                                                  **decodeparame)


    def forward(self, condition, infer=False):

        return self.model(condition,infer)

    def get_loss(self):
        return self.model.build_loss()


