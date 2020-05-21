import random
import timeit
from pathlib import Path
from pickle import dump

import nltk
from nltk.tag.brill import brill24
from nltk.tag.brill_trainer import BrillTaggerTrainer

ud_pos = {'shouldSplit': False, 'tagIndex': 3, 'name': 'ud_pos', 'patterns': [
    (r'.*ki$', 'ADJ'),
    (r'.*ti$', 'VERB'),
    (r'.*ći$', 'VERB'),
    (r'^-?[0-9]+(\.[0-9]+)?$', 'NUM'),
    (r'.*', 'NOUN')
]}
smd_pos = {'shouldSplit': True, 'tagIndex': 2, 'name': 'smd_pos', 'patterns': [
    (r'.*ki$', 'A'),
    (r'.*ti$', 'V'),
    (r'.*ći$', 'V'),
    (r'^-?[0-9]+(\.[0-9]+)?$', 'NUM'),
    (r'.*', 'N')
]}
n_pos = {'shouldSplit': False, 'tagIndex': 2, 'name': 'n_pos', 'patterns': [
    (r'.*ki$', 'A:m'),
    (r'.*ti$', 'V'),
    (r'.*ći$', 'V'),
    (r'^-?[0-9]+(\.[0-9]+)?$', 'NUM'),
    (r'.*', 'V')
]}

TAGGER_LIST = ['DEFAULT', 'REGEXP', 'LOOKUP', 'UNIGRAM', 'AFFIX', 'BIGRAM', 'TRIGRAM', 'CRF', 'HMM', 'PERCEPTRON',
               'TNT']
TOP_TAGGER_LIST = ['CRF', 'CRF-UNSEEN', 'CRF-BRILL', 'CRF-BRILL-UNSEEN', 'PERCEPTRON', 'PERCEPTRON-UNSEEN',
                   'PERCEPTRON-BRILL', 'PERCEPTRON-BRILL-UNSEEN']

CONSTANT_SEED_FOR_RANDOM = 92347816


def parse_sentences(input_data, tagset):
    input_tagged_sents = []
    last_sentence = []
    last_sentence_number = 1
    for idx in range(len(input_data)):  # iterate through every word
        parts = input_data[idx].split()
        sentence_number = int(parts[0])  # 1
        token = parts[1]  # Jaroslav
        if tagset['shouldSplit']:
            tag = parts[tagset['tagIndex']].split(':')[0]
        else:
            tag = parts[tagset['tagIndex']]

        if sentence_number != last_sentence_number:  # if this is the end of the sentence, finalize it and build a new one
            last_sentence_number = sentence_number
            input_tagged_sents.append(last_sentence)
            last_sentence = []

        last_sentence.append((token, tag))

    # add the last sentence
    input_tagged_sents.append(last_sentence)

    return input_tagged_sents


def strip_tags(original_set):
    stripped_set = []
    for tagged_sent in original_set:
        sent_stripped = [token for (token, tag) in tagged_sent]
        stripped_set.append(sent_stripped)
    return stripped_set


def split_data_set(input_tagged_sents):
    random.seed(CONSTANT_SEED_FOR_RANDOM)
    random.shuffle(input_tagged_sents)
    train_slices = []
    test_slices = []
    size = len(input_tagged_sents)

    for k in range(0, 10):
        train_set = []
        if k > 0:
            train_set.extend(input_tagged_sents[:int(size * k / 10)])
        if k < 9:
            train_set.extend(input_tagged_sents[int(size * (k + 1) / 10):])
        train_slices.append(train_set)

        test_set = input_tagged_sents[int(size * k / 10):int(size * (k + 1) / 10)]
        test_slices.append(test_set)

    return train_slices, test_slices


def score(predicted_slice, gold_standard, scores):
    for sent_idx in range(0, len(predicted_slice)):
        # extract predicted and actual sentences
        predicted_sent = predicted_slice[sent_idx]
        gold_sent = gold_standard[sent_idx]

        # iterate through sentences and score each prediction
        for token_idx in range(0, len(predicted_sent)):

            # extract predicted and actual tag
            predicted_tag = predicted_sent[token_idx][1]
            actual_tag = gold_sent[token_idx][1]

            # add tags to the scores if they don't exist yet
            if predicted_tag not in scores:
                scores[predicted_tag] = {'tp': 0, 'fp': 0, 'fn': 0}
            if actual_tag not in scores:
                scores[actual_tag] = {'tp': 0, 'fp': 0, 'fn': 0}

            # resolve between True Positive case, False Positive case and False Negative. We need those for P, R and F1.
            if predicted_tag == actual_tag:
                scores[predicted_tag]['tp'] += 1
            else:
                scores[predicted_tag]['fp'] += 1
                scores[actual_tag]['fn'] += 1


def evaluate_tagger(tagger, test_set, scores_tagger):
    evaluation = {'accuracy': tagger.evaluate(test_set)}

    predicted_set = tagger.tag_sents(strip_tags(test_set))
    evaluation['tags'] = {}
    score(predicted_set, test_set, evaluation['tags'])

    scores_tagger.append(evaluation)


def calculate_most_frequent_tag(input_tagged_sents):
    tokens = [pair[1] for sent in input_tagged_sents for pair in sent]
    tag_fd = nltk.FreqDist(tokens)
    return tag_fd.most_common()[0][0]


def get_most_likely_tags(train_set):
    tagged_tokens = [pair for sent in train_set for pair in sent]
    cfd = nltk.ConditionalFreqDist(tagged_tokens)

    tokens = [pair[0] for pair in tagged_tokens]
    most_freq_tokens = nltk.FreqDist(tokens).most_common(100)

    likely_tags = dict((token, cfd[token].max()) for (token, _) in most_freq_tokens)
    return likely_tags


def train_all_taggers(train_set, test_set, scores, patterns):
    # Default tagger
    default_tagger = nltk.DefaultTagger(calculate_most_frequent_tag(train_set))
    evaluate_tagger(default_tagger, test_set, scores['DEFAULT'])
    # RegExp tagger
    regexp_tagger = nltk.RegexpTagger(patterns)
    evaluate_tagger(regexp_tagger, test_set, scores['REGEXP'])
    # Lookup tagger
    lookup_tagger = nltk.UnigramTagger(model=get_most_likely_tags(train_set))
    evaluate_tagger(lookup_tagger, test_set, scores['LOOKUP'])
    # Unigram tagger
    unigram_tagger = nltk.UnigramTagger(train_set, backoff=default_tagger)
    evaluate_tagger(unigram_tagger, test_set, scores['UNIGRAM'])
    # Affix tagger
    affix_tagger = nltk.AffixTagger(train_set, backoff=unigram_tagger)
    evaluate_tagger(affix_tagger, test_set, scores['AFFIX'])
    # Bigram tagger
    bigram_tagger = nltk.BigramTagger(train_set, backoff=affix_tagger)
    evaluate_tagger(bigram_tagger, test_set, scores['BIGRAM'])
    # Trigram tagger
    trigram_tagger = nltk.TrigramTagger(train_set, backoff=bigram_tagger)
    evaluate_tagger(trigram_tagger, test_set, scores['TRIGRAM'])
    # CRF tagger
    crf_tagger = nltk.tag.CRFTagger()
    crf_tagger.train(train_set, 'model.crf.tagger')
    evaluate_tagger(crf_tagger, test_set, scores['CRF'])
    # HMM tagger
    hmm_tagger = nltk.tag.hmm.HiddenMarkovModelTrainer().train_supervised(train_set)
    evaluate_tagger(hmm_tagger, test_set, scores['HMM'])
    # Perceptron tagger
    perceptron_tagger = nltk.tag.perceptron.PerceptronTagger(load=False)
    perceptron_tagger.train(train_set, nr_iter=10)
    evaluate_tagger(perceptron_tagger, test_set, scores['PERCEPTRON'])
    # TnT tagger
    tnt_tagger = nltk.tag.tnt.TnT(unk=unigram_tagger, Trained=True)
    tnt_tagger.train(train_set)
    evaluate_tagger(tnt_tagger, test_set, scores['TNT'])


def crossvalidate_taggers(train_slices, test_slices, patterns):
    scores = {name: [] for name in TAGGER_LIST}
    start = timeit.default_timer()
    print('Started training and evaluation of taggers.')
    for k in range(0, 10):
        train_all_taggers(train_slices[k], test_slices[k], scores, patterns)
        print('End of the epoch {}/9. Time from the start: {}'.format(k, timeit.default_timer() - start))
    print('Training and evaluation finished.')
    return scores


def print_total_tag_scores(tagger, tag_total):
    precision = tag_total['tp'] / (tag_total['tp'] + tag_total['fp'])
    recall = tag_total['tp'] / (tag_total['tp'] + tag_total['fn'])
    f1 = 2 * precision * recall / (precision + recall)
    tag_total_scores = {'precision': precision, 'recall': recall, 'f1': f1}
    print("Tagger: {}. Scores: {}".format(tagger, tag_total_scores))


def average_tag(scores, tagger_list=TAGGER_LIST):
    average_tag_metrics = []
    for tagger in tagger_list:
        tag_sum = {}
        iteration_count = len(scores[tagger])
        tag_sum_total = {'tp': 0, 'fp': 0, 'fn': 0}
        for i in range(0, iteration_count):
            for tag, metrics in scores[tagger][i]['tags'].items():
                if tag not in tag_sum:
                    tag_sum[tag] = {'tp': 0, 'fp': 0, 'fn': 0}
                tag_sum[tag]['tp'] += metrics['tp']
                tag_sum[tag]['fp'] += metrics['fp']
                tag_sum[tag]['fn'] += metrics['fn']

        tag_avg = {}
        for tag, metrics in tag_sum.items():
            tp = tag_sum[tag]['tp'] / iteration_count
            fp = tag_sum[tag]['fp'] / iteration_count
            fn = tag_sum[tag]['fn'] / iteration_count
            tag_sum_total['tp'] += tp
            tag_sum_total['fp'] += fp
            tag_sum_total['fn'] += fn
            if tp + fp == 0 or tp + fn == 0:
                precision = None
                recall = None
                f1 = None
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                if precision + recall == 0:
                    f1 = None
                else:
                    f1 = 2 * precision * recall / (precision + recall)
            tag_avg[tag] = {'precision': precision, 'recall': recall, 'f1': f1}
        print_total_tag_scores(tagger, tag_sum_total)
        average_tag_metrics.append(tag_avg)
    return average_tag_metrics


def average(scores, tagger_list=TAGGER_LIST):
    average_scores = []
    for tagger in tagger_list:
        tagger_sum = 0
        iteration_count = len(scores[tagger])
        if iteration_count == 0:
            continue
        for i in range(0, iteration_count):
            tagger_sum += scores[tagger][i]['accuracy']
        average_scores.append(tagger_sum / iteration_count)
    return average_scores


def save_tagger(model, name):
    # saves a tagger model to a file
    output = open(name, 'wb')
    dump(model, output, -1)
    output.close()


def load_tagger(model, name):
    # use this to load tagger from file
    output = open(name, 'wb')
    dump(model, output, -1)
    output.close()


def train_crf_and_perceptron(train_set, test_set, unseen_set, scores):
    # CRF tagger
    crf_tagger = nltk.tag.CRFTagger()
    crf_tagger.train(train_set, 'model.crf.tagger')
    evaluate_tagger(crf_tagger, test_set, scores['CRF'])
    evaluate_tagger(crf_tagger, unseen_set, scores['CRF-UNSEEN'])
    # CRF tagger improved with Brill
    crf_tagger_brill = BrillTaggerTrainer(crf_tagger, brill24()).train(train_set)
    evaluate_tagger(crf_tagger_brill, test_set, scores['CRF-BRILL'])
    evaluate_tagger(crf_tagger_brill, unseen_set, scores['CRF-BRILL-UNSEEN'])
    # Perceptron tagger
    perceptron_tagger = nltk.tag.perceptron.PerceptronTagger(load=False)
    perceptron_tagger.train(train_set, nr_iter=10)
    save_tagger(perceptron_tagger, "perceptron.pkl")
    evaluate_tagger(perceptron_tagger, test_set, scores['PERCEPTRON'])
    evaluate_tagger(perceptron_tagger, unseen_set, scores['PERCEPTRON-UNSEEN'])
    # Perceptron tagger improved with Brill
    perceptron_tagger_brill = BrillTaggerTrainer(perceptron_tagger, brill24()).train(train_set)
    evaluate_tagger(perceptron_tagger_brill, test_set, scores['PERCEPTRON-BRILL'])
    evaluate_tagger(perceptron_tagger_brill, unseen_set, scores['PERCEPTRON-BRILL-UNSEEN'])
    pass


def benchmark_crf_and_perceptron(train_slices, test_slices, unseen_slice):
    scores = {name: [] for name in TOP_TAGGER_LIST}
    start = timeit.default_timer()
    print('Started benchmarking CRF and Perceptron. ')
    for k in range(0, 10):
        train_crf_and_perceptron(train_slices[k], test_slices[k], unseen_slice, scores)
        print('End of the epoch {}/9. Time from the start: {}'.format(k, timeit.default_timer() - start))
    print('Benchmarking CRF and Perceptron finished.')
    return scores


def output(avg_tagger_accuracy, avg_tag_metrics, tagset, name_suffix):
    output_text = ''
    for tagger_idx, accuracy in enumerate(avg_tagger_accuracy):
        output_text += '{} tagger: {}\n'.format(TOP_TAGGER_LIST[tagger_idx], accuracy)
        for avg_tag in avg_tag_metrics[tagger_idx].items():
            output_text += '{} {}\n'.format(avg_tag[0], avg_tag[1])
        output_text += '\n\n'

    Path('{}_{}.txt'.format(tagset['name'], name_suffix)).write_text(output_text, encoding='utf-8-sig')


def evaluate_all_taggers():
    for tagset in [ud_pos, smd_pos, n_pos]:
        input_tagged_sents = parse_sentences(input_data, tagset)
        train_slices, test_slices = split_data_set(input_tagged_sents)

    scores = crossvalidate_taggers(train_slices, test_slices, tagset['patterns'])

    avg_tagger_accuracy = average(scores)
    avg_tag_metrics = average_tag(scores)

    print(avg_tagger_accuracy)
    print(avg_tag_metrics)
    output(avg_tagger_accuracy, avg_tag_metrics, tagset, 'all_taggers')


def evaluate_top_taggers():
    tagset = smd_pos # Here you select tagset for which you want to evalute top taggers
    input_tagged_sents = parse_sentences(input_data, tagset)
    train_slices, test_slices = split_data_set(input_tagged_sents)

    unseen_text = Path('unseen_text.txt').read_text(encoding="utf-8-sig").strip().splitlines()
    unseen_tagged_sents = parse_sentences(unseen_text, tagset)

    top_tagger_scores = benchmark_crf_and_perceptron(train_slices, test_slices, unseen_tagged_sents)
    print(top_tagger_scores)

    avg_tagger_accuracy = average(top_tagger_scores, TOP_TAGGER_LIST)
    avg_tag_metrics = average_tag(top_tagger_scores, TOP_TAGGER_LIST)

    output(avg_tagger_accuracy, avg_tag_metrics, tagset, 'crf_and_perceptron')


if __name__ == '__main__':
    input_data = Path('tagged-data-original.txt').read_text(encoding='utf-8-sig').strip().splitlines()

    evaluate_all_taggers()
    evaluate_top_taggers()
