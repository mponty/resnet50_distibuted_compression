import torch
import numpy as np
from scipy import ndimage
from poutyne import torch_to_numpy, numpy_to_torch


def fast_quantile_encode(weight, window_size=4.):
    scale = 128 / (window_size * weight.std())

    scaled_weight = scale * weight
    quant_weight = scaled_weight.astype('int8')

    quant_weight[scaled_weight > 127] = 127
    quant_weight[scaled_weight < -128] = -128

    quant_weight = quant_weight.astype('uint8')

    lookup = ndimage.mean(weight, labels=quant_weight, index=np.arange(256))
    lookup[np.isnan(lookup)] = 0.
    lookup = lookup.astype('float32')
    return quant_weight, lookup


UNIFORM_BUCKETS_STD_RANGE = 6
UINT8_RANGE = 256


def uint8_uniform_buckets_encode(tensor: torch.Tensor, range_in_sigmas: float):
    offset = UINT8_RANGE // 2
    shift = tensor.mean()
    scale = range_in_sigmas * tensor.std() / UINT8_RANGE

    quant_weight = torch.quantize_per_tensor(tensor - shift, scale, offset, torch.quint8).int_repr()

    lookup = torch.arange(0, UINT8_RANGE, dtype=torch.float32, device=tensor.device)
    lookup = scale * (lookup - offset) + shift
    # Take into account the tails of distribution
    lookup[0], lookup[-1] = tensor[quant_weight == 0].mean(), tensor[quant_weight == UINT8_RANGE - 1].mean()

    return quant_weight, lookup


class UINT8Compressor(object):
    def __init__(self, parameter_names):
        self.parameter_names = parameter_names

    def encode(self, weight):
        with torch.no_grad():
            quant_weight, lookup = uint8_uniform_buckets_encode(weight, UNIFORM_BUCKETS_STD_RANGE)
            return dict(quant_weight=quant_weight, lookup=lookup)

    def decode(self, encoded):
        quant_weight, lookup = encoded['quant_weight'], encoded['lookup']
        return lookup[quant_weight.long()].float()

    def serialize(self, state_dict):
        for name in self.parameter_names:
            state_dict[name] = self.encode(state_dict[name])
        return state_dict

    def deserialize(self, state_dict):
        for name in self.parameter_names:
            state_dict[name] = self.decode(state_dict[name])
        return state_dict


class UINT8MinMaxCompressor(object):
    def __init__(self, parameter_names, shift=128):
        self.parameter_names = parameter_names
        self.shift = shift

    def encode(self, weight):
        with torch.no_grad():
            scale = (weight.max() - weight.min()) / 256
            quant_weight = torch.quantize_per_tensor(weight, scale, self.shift, torch.quint8).int_repr()
            return dict(quant_weight=quant_weight, scale=scale)

    def decode(self, encoded):
        quant_weight, scale = encoded['quant_weight'], encoded['scale']
        return scale * (quant_weight.float() - self.shift)

    def serialize(self, state_dict):
        for name in self.parameter_names:
            state_dict[name] = self.encode(state_dict[name])
        return state_dict

    def deserialize(self, state_dict):
        for name in self.parameter_names:
            state_dict[name] = self.decode(state_dict[name])
        return state_dict
