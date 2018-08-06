#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse

import ogan.opts as opts
from ogan.train_single import main as single_main
from ogan.utils.logging import init_logger


def main(opt):
    init_logger(opt.log_file)

    if opt.epochs:
        raise AssertionError("-epochs is deprecated please use -train_steps.")

    if len(opt.gpuid) == 1:
        single_main(opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py')

    opts.embedding_opts(parser)
    opts.memory_opts(parser)
    opts.discriminator_opts(parser)
    opts.generator_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)