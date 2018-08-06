import os
import torch
import torch.nn as nn

import ogan.inputters

from collections import deque
from shutil import copyfile
from ogan.utils.logging import logger


def build_model_saver(embedding_opt, memory_opt, discriminator_opt, generator_opt, opt,
                      memory, discriminator, generator,
                      fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             memory, discriminator, generator,
                             embedding_opt, memory_opt, discriminator_opt, generator_opt,
                             fields, optim,
                             opt.save_checkpoint_steps, opt.keep_checkpoint)
    return model_saver


class ModelSaverBase(object):
    """
        Base class for model saving operations
        Inherited classes must implement private methods:
            * `_save`
            * `_rm_checkpoint
    """

    def __init__(self, base_path, memory, discriminator, generator,
                 embedding_opt, memory_opt, discriminator_opt, generator_opt,
                 fields, optim,
                 save_checkpoint_steps, keep_checkpoint=-1):
        self.base_path = base_path
        self.memory = memory
        self.discriminator = discriminator
        self.generator = generator
        self.embedding_opt = embedding_opt
        self.memory_opt = memory_opt
        self.discriminator_opt = discriminator_opt
        self.generator_opt = generator_opt
        self.fields = fields
        self.optim = optim
        self.keep_checkpoint = keep_checkpoint
        self.save_checkpoint_steps = save_checkpoint_steps

        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def maybe_save(self, step):
        """
        Main entry point for model saver
        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """
        if self.keep_checkpoint == 0:
            return

        if step % self.save_checkpoint_steps != 0:
            return

        chkpt, chkpt_name = self._save(step)

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step):
        """ Save a resumable checkpoint.
        Args:
            step (int): step number
        Returns:
            checkpoint: the saved object
            checkpoint_name: name (or path) of the saved checkpoint
        """
        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """
        Remove a checkpoint
        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        raise NotImplementedError()

    def _cp_checkpoint(self, src_file_name, tgt_file_name):
        """
        copy a checkpoint as the best checkpoint
        Args:
            src_file_name(str): name that indentifies the checkpoint
                (it may be a filepath)
            tgt_file_name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """
        Simple model saver to filesystem
    """

    def __init__(self, base_path, memory, discriminator, generator,
                 embedding_opt, memory_opt, discriminator_opt, generator_opt,
                 fields, optim,
                 save_checkpoint_steps, keep_checkpoint=0):
        super(ModelSaver, self).__init__(base_path, memory, discriminator, generator,
                 embedding_opt, memory_opt, discriminator_opt, generator_opt,
                 fields, optim,
                 save_checkpoint_steps, keep_checkpoint)

    def _save(self, step):
        discriminator_state_dict = self.discriminator.state_dict()
        generator_state_dict = self.generator.state_dict()
        checkpoint = {
            'discriminator': discriminator_state_dict,
            'generator': generator_state_dict,
            'vocab': ogan.inputters.save_fields_to_vocab(self.fields),
            'embedding_opt': self.embedding_opt,
            'memory_opt': self.memory_opt,
            'discriminator_opt': self.discriminator_opt,
            'generator_opt': self.generator_opt,
            'discriminator_optim': self.optim,
            'generator_optim': self.optim
        }

        logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)

    def _cp_checkpoint(self, src_file_name, tgt_file_name):
        copyfile(src_file_name, tgt_file_name)