#!/usr/bin/env python3

"""calculate BLEU scores
script taken from https://github.com/vikasnar/Bleu
and adjusted by Jörg Tiedemann
"""


import sys
import codecs
import os
import math
import operator
import json
import functools


def fetch_data(cand, ref):
    """ Store each reference and candidate sentences as a list """
    references = []
    if os.path.isdir(ref):
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'utf-8')
                references.append(reference_file.readlines())
    else:
        reference_file = codecs.open(ref, 'r', 'utf-8')
        references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    return candidate, references


def count_ngram(candidate, references, n, lowercase):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n])
                if lowercase:
                    ngram = ngram.lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n])
            if lowercase:
                ngram = ngram.lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    elif c == 0:
        bp = 0
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (functools.reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(candidate, references, lowercase=False):
    precisions = []
    for i in range(4):
        pr, bp = count_ngram(candidate, references, i+1, lowercase)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp
    return bleu, precisions[0], precisions[1], precisions[2], precisions[3], bp

if __name__ == "__main__":
    candidate, references = fetch_data("/home/zeng/sciences/openGAN/data/test/a.txt",
                                       "/home/zeng/sciences/openGAN/data/test/b.txt")
    bleu = BLEU(candidate, references)
    print('BLEU = %.4f (%.3f, %.3f, %.3f, %.3f, BP = %.3f)' % (bleu))

    # from nltk.translate.bleu_score import sentence_bleu
    #
    # reference = [['The', 'cat', 'is', 'on', 'the', 'mat']]
    # candidate = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    # score = sentence_bleu(reference, candidate)
    # print(score)
