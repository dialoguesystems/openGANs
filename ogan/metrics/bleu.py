# -*- coding: utf-8 -*-
"""
From https://github.com/ngohoanhkhoa/GAN-NMT/blob/master/nmtpy/nmtpy/metrics/bleu.py
"""
import subprocess
import pkg_resources
import copy
import sys, math, re
from collections import defaultdict

from ogan.metrics import MetricBase

BLEU_SCRIPT = pkg_resources.resource_filename("ogan", "metrics/multi-bleu.perl")

class BLEUScore(MetricBase):
    def __init__(self, score=None):
        super(BLEUScore, self).__init__(score)
        self.name = "BLEU"
        if score:
            self.score = float(score.split()[2][:-1])
            self.score_str = score.replace('BLEU = ', '')

"""MultiBleuScorer class."""
class MultiBleuScorer(object):
    def __init__(self, lowercase=False):
        # For multi-bleu.perl we give the reference(s) files as argv,
        # while the candidate translations are read from stdin.
        self.lowercase = lowercase
        self.__cmdline = [BLEU_SCRIPT]
        if self.lowercase:
            self.__cmdline.append("-lc")

    def compute(self, refs, hypfile):
        cmdline = self.__cmdline[:]

        # Make reference files a list
        refs = [refs] if isinstance(refs, str) else refs
        cmdline.extend(refs)

        hypstring = None
        with open(hypfile, "r") as fhyp:
            hypstring = fhyp.read().rstrip()

        score = subprocess.run(cmdline, stdout=subprocess.PIPE,
                               input=hypstring, universal_newlines=True).stdout.splitlines()
        if len(score) == 0:
            return BLEUScore()
        else:
            return BLEUScore(score[0].rstrip("\n"))


if __name__ == '__main__':
    multi_bleu_scorer = MultiBleuScorer()
    score = multi_bleu_scorer.compute("/home/zeng/sciences/openGAN/data/dailydialog/single_turn/test.src", "/home/zeng/sciences/openGAN/data/dailydialog/single_turn/test.tgt")
    print(score)