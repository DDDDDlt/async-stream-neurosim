import os
import torch
import torch.nn as nn

from modules.async_stream_infer import AsyncStreamConv2d, AsyncStreamLinear


class StreamCNN(nn.Module):
    def __init__(self, args, logger):
        super(StreamCNN, self).__init__()
        self.features = nn.Sequential(
            AsyncStreamConv2d(
                1, 8, kernel_size=3, logger=logger, wl_input=args.wl_activate,
                wl_weight=args.wl_weight, inference=args.inference,
                stream_frequency=args.stream_frequency, stream_window=args.stream_window,
                stream_jitter=args.stream_jitter, name="Conv0_"
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            AsyncStreamConv2d(
                8, 16, kernel_size=3, logger=logger, wl_input=args.wl_activate,
                wl_weight=args.wl_weight, inference=args.inference,
                stream_frequency=args.stream_frequency, stream_window=args.stream_window,
                stream_jitter=args.stream_jitter, name="Conv1_"
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            AsyncStreamLinear(
                16 * 5 * 5, 64, logger=logger, wl_input=args.wl_activate,
                wl_weight=args.wl_weight, inference=args.inference,
                stream_frequency=args.stream_frequency, stream_window=args.stream_window,
                stream_jitter=args.stream_jitter, name="FC0_"
            ),
            nn.ReLU(),
            AsyncStreamLinear(
                64, 10, logger=logger, wl_input=args.wl_activate,
                wl_weight=args.wl_weight, inference=args.inference,
                stream_frequency=args.stream_frequency, stream_window=args.stream_window,
                stream_jitter=args.stream_jitter, name="FC1_"
            ),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def streamcnn(args, logger, pretrained=None):
    model = StreamCNN(args=args, logger=logger)
    if pretrained is not None and os.path.exists(pretrained):
        model.load_state_dict(torch.load(pretrained, map_location="cpu"))
    return model
