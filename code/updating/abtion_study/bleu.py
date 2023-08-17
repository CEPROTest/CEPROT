
import collections
import math
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from sklearn.metrics import consensus_score

def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0
  if reference_length != 0:
    ratio = float(translation_length) / reference_length
  else:
    ratio = 0

  if ratio > 1.0:
    bp = 1.
  elif ratio == 0:
    bp = 0
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)


def _bleu(ref_file, trans_file, subword_option=None):
    max_order = 4
    smooth = True
    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with open(reference_filename) as fh:
            reference_text.append(fh.readlines())
    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference_list.append(reference.strip().split())
        per_segment_references.append(reference_list)
    translations = []
    with open(trans_file) as fh:
        for line in fh:
            translations.append(line.strip().split())
    bleu_score, _, _, _, _, _ = compute_bleu(per_segment_references, translations, max_order, smooth)
    return round(100 * bleu_score,2)

def compute_sentence_level_blue(ref_file, trans_file, subword_option=None):
    smooth = SmoothingFunction()
    ref = []
    with open(ref_file) as fh:
        for line in fh:
            ref.append(line.strip().split())

    translations = []
    with open(trans_file) as fh:
        for line in fh:
            translations.append(line.strip().split())
    sentence_scorces = []
    if len(ref) == len(translations):
      for i in range(len(ref)):
        s_score = 100*round(sentence_bleu([ref[i]], translations[i],smoothing_function=smooth.method1),4)
        sentence_scorces.append(s_score)
    return sentence_scorces
        