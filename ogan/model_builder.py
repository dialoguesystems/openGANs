"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import ogan.inputters as inputters
import ogan.modules

from ogan.discriminator import LSTM_Discriminator, CNN_Discriminator
from ogan.generator import LSTM_Generator
from ogan.memory import LSTM_Memory

from ogan.modules import Embeddings
from ogan.utils.misc import use_gpu
from ogan.utils.logging import logger


def build_embeddings(opt, word_dict, for_source=True):
    """
    Build an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        for_source(bool): build Embeddings for source or target language?
    """
    if for_source:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[inputters.PAD_WORD]
    num_word_embeddings = len(word_dict)

    return Embeddings(word_vocab_size=num_word_embeddings,
                      word_vec_size=embedding_dim,
                      word_padding_idx=word_padding_idx,
                      position_encoding=opt.position_encoding,
                      dropout=opt.pe_dropout)


def build_memory(opt, embeddings):
    return LSTM_Memory(embeddings, opt.memory_rnn_size, bidirectional=True)


def build_discriminator(opt, embeddings, memory=None):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.discriminator_model_type == "cnn":
        return CNN_Discriminator(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    else:
        return LSTM_Discriminator(embeddings, opt.discriminator_rnn_size, opt.discriminator_dropout)


def build_generator(opt, embeddings, memory=None):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
        memory: conditional inputs | the "encoder" in traditional seq2seq model. |
    """
    if opt.generator_model_type == "lstm":
        return LSTM_Generator(opt.generator_rnn_size, embeddings, memory)



def load_test_model(opt, dummy_opt, best=True):
    """ Load model for Inference """
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    fields = inputters.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(embedding_opt, memory_opt, discriminator_opt, generator_opt, fields, gpu, checkpoint=None):
    """
    Args:
        discriminator_opt: the option loaded from discriminator checkpoint.
        generator_opt: the option loaded from generator checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the Discriminator & Generator.
    """

    # Build embeddings for source & target language.
    src_dict = fields["src"].vocab
    src_embeddings = build_embeddings(embedding_opt, src_dict)
    tgt_dict = fields["tgt"].vocab
    tgt_embeddings = build_embeddings(embedding_opt, tgt_dict, for_source=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if embedding_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight


    # Build Conditional Encoder
    memory = build_memory(memory_opt, src_embeddings)


    # Build Discriminator.
    discriminator = build_discriminator(discriminator_opt, tgt_embeddings, memory)


    # Build Generator.
    generator = build_generator(generator_opt, tgt_embeddings, memory)


    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        memory.load_state_dict(checkpoint['memory'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        def init_params(model, model_opt):
            if model_opt.param_init != 0.0:
                for p in model.parameters():
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            if model_opt.param_init_glorot:
                for p in model.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
        init_params(memory, memory_opt)
        init_params(discriminator, discriminator_opt)
        init_params(generator, generator_opt)


        if hasattr(discriminator, 'src_embeddings'):
            discriminator.src_embeddings.load_pretrained_vectors(
                discriminator_opt.pre_word_vecs_enc, discriminator_opt.fix_word_vecs_enc)
        if hasattr(discriminator, 'tgt_embeddings'):
            discriminator.tgt_embeddings.load_pretrained_vectors(
                discriminator_opt.pre_word_vecs_enc, discriminator_opt.fix_word_vecs_enc)
        if hasattr(generator, 'src_embeddings'):
            generator.src_embeddings.load_pretrained_vectors(
                generator_opt.pre_word_vecs_dec, generator_opt.fix_word_vecs_dec)
        if hasattr(generator, 'tgt_embeddings'):
            generator.tgt_embeddings.load_pretrained_vectors(
                generator_opt.pre_word_vecs_dec, generator_opt.fix_word_vecs_dec)


    device = torch.device("cuda" if gpu else "cpu")
    memory.to(device)
    discriminator.to(device)
    generator.to(device)
    torch.device("cuda" if gpu else "cpu")
    return memory, discriminator, generator


def build_model(embedding_opt, memory_opts, discriminator_opt, generator_opt, opt, fields, checkpoint):
    """ Build the discriminator & generator """
    logger.info('Building memory & discriminator & generator ...')
    memory, discriminator, generator = build_base_model(embedding_opt, memory_opts,  discriminator_opt, generator_opt, fields, use_gpu(opt), checkpoint)
    logger.info(memory)
    logger.info(discriminator)
    logger.info(generator)

    return memory, discriminator, generator