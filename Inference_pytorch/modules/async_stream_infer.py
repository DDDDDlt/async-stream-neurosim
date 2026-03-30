import torch
import torch.nn as nn
import torch.nn.functional as F

from utee import async_stream


class AsyncStreamConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, logger=None, wl_input=8,
                 wl_weight=8, inference=0, stream_frequency=1e4,
                 stream_window=32.0, stream_jitter=0.02, stream_min_precision=4.0,
                 stream_max_precision=12.0, name="AsyncConv"):
        super(AsyncStreamConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.logger = logger
        self.wl_input = wl_input
        self.wl_weight = wl_weight
        self.inference = inference
        self.stream_frequency = stream_frequency
        self.stream_window = stream_window
        self.stream_jitter = stream_jitter
        self.stream_min_precision = stream_min_precision
        self.stream_max_precision = stream_max_precision
        self.name = name
        self.last_stream_stats = None

    def forward(self, input):
        if not self.inference:
            self.last_stream_stats = None
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        stats = async_stream.compute_stream_stats(
            input, self.stream_frequency, self.stream_window, self.stream_jitter,
            min_precision=self.stream_min_precision, max_precision=self.stream_max_precision
        )
        self.last_stream_stats = stats

        stream_input = async_stream.quantize_tensor(input, stats["effective_precision"])
        stream_input = async_stream.apply_stream_noise(stream_input, stats, self.stream_jitter)
        weight = async_stream.quantize_tensor(self.weight, self.wl_weight) if self.inference else self.weight
        return F.conv2d(stream_input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class AsyncStreamLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, logger=None,
                 wl_input=8, wl_weight=8, inference=0, stream_frequency=1e4,
                 stream_window=32.0, stream_jitter=0.02, stream_min_precision=4.0,
                 stream_max_precision=12.0, name="AsyncLinear"):
        super(AsyncStreamLinear, self).__init__(in_features, out_features, bias)
        self.logger = logger
        self.wl_input = wl_input
        self.wl_weight = wl_weight
        self.inference = inference
        self.stream_frequency = stream_frequency
        self.stream_window = stream_window
        self.stream_jitter = stream_jitter
        self.stream_min_precision = stream_min_precision
        self.stream_max_precision = stream_max_precision
        self.name = name
        self.last_stream_stats = None

    def forward(self, input):
        if not self.inference:
            self.last_stream_stats = None
            return F.linear(input, self.weight, self.bias)

        stats = async_stream.compute_stream_stats(
            input, self.stream_frequency, self.stream_window, self.stream_jitter,
            min_precision=self.stream_min_precision, max_precision=self.stream_max_precision
        )
        self.last_stream_stats = stats

        stream_input = async_stream.quantize_tensor(input, stats["effective_precision"])
        stream_input = async_stream.apply_stream_noise(stream_input, stats, self.stream_jitter)
        weight = async_stream.quantize_tensor(self.weight, self.wl_weight) if self.inference else self.weight
        return F.linear(stream_input, weight, self.bias)
