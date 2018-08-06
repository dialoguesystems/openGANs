#!/usr/bin/env python
"""
    Training on a single process
"""
from __future__ import division

import argparse
import os
import random
import torch

import ogan.opts as opts

from ogan.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields
from ogan.model_builder import build_model
from ogan.utils.optimizers import build_optim
from ogan.trainer import build_trainer
from ogan.model_saver import build_model_saver
from ogan.utils.logging import logger


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_postprocessing(opt):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if torch.cuda.is_available() and not opt.gpuid:
        logger.info("WARNING: You have a CUDA device, should run with -gpuid")

    if opt.gpuid:
        torch.cuda.set_device(opt.device_id)
        if opt.seed > 0:
            # this one is needed for torchtext random call (shuffled iterator)
            # in multi gpu it ensures datasets are read in the same order
            random.seed(opt.seed)
            # These ensure same initialization in multi gpu mode
            torch.manual_seed(opt.seed)
            torch.cuda.manual_seed(opt.seed)

    return opt


def main(opt):
    opt = training_opt_postprocessing(opt)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        embedding_opt = checkpoint["embedding_opt"]
        memory_opt = checkpoint["memory_opt"]
        discriminator_opt = checkpoint['discriminator_opt']
        generator_opt = checkpoint['generator_opt']

    else:
        checkpoint = None
        embedding_opt = memory_opt = discriminator_opt = generator_opt = opt

    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", opt))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, opt, checkpoint)


    # Build discriminator & generator.
    # discriminator = build_discriminator(discriminator_opt, opt, fields, discriminator_checkpoint)
    # generator = build_generator(generator_opt, opt, fields, generator_checkpoint)
    memory, discriminator, generator = build_model(embedding_opt, memory_opt, discriminator_opt, generator_opt, opt, fields, checkpoint)

    # n_params, enc, dec = _tally_parameters(discriminator)
    # n_params, enc, dec = _tally_parameters(generator)
    # logger.info('encoder: %d' % enc)
    # logger.info('decoder: %d' % dec)
    # logger.info('* number of parameters: %d' % n_params)
    # _check_save_model_path(opt)


    # Build optimizer.
    discriminator_optim = build_optim([discriminator, memory], opt, checkpoint)
    generator_optim = build_optim([generator, memory], opt, checkpoint)


    # Build model saver
    model_saver = build_model_saver(embedding_opt, memory_opt, discriminator_opt, generator_opt, opt,
                                    memory, discriminator, generator,
                                    fields, discriminator_optim)


    trainer = build_trainer(opt, memory, discriminator, generator, fields, discriminator_optim, generator_optim, data_type,
                            model_saver=model_saver)

    def train_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("train", opt), fields, opt)

    def valid_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("valid", opt), fields, opt)

    # Do training.
    trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps,
                  opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py')

    opts.embedding_opts(parser)
    opts.memory_opts(parser)
    opts.discriminator_opts(parser)
    opts.generator_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)