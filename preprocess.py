import argparse

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from data_utils import TextMelLoader, TextMelCollate
from hparams import create_hparams
from train import prepare_dataloaders


def preprocess(hparams):

    train_loader, valset, testset, _ = prepare_dataloaders(hparams)
    
    if hparams.save_mel_to_disk:
        for _ in train_loader:
            pass
        for _ in valset:
            pass
        for _ in testset:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    preprocess(hparams)
