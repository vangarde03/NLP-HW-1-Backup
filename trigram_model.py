import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    result = []
    if n < 1:
        return result
    start = 0-n+1
    end = start+n-1
    while end <= len(sequence):
        ngram = []
        for i in range(start, end+1):
            if n == 1 and start <= 0:  # base case for unigram
                result.append(tuple(["START"]))
            if i < 0:
                ngram.append("START")
                continue
            if i >= len(sequence):
                ngram.append("STOP")
                continue
            ngram.append(sequence[i])
        result.append(tuple(ngram))
        start += 1
        end += 1
    return result


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        generator = corpus_reader(corpusfile)
        self.total_words = sum(len(sentence) for sentence in generator)

        # Part 5 Testing
        # generator = corpus_reader(corpusfile)  # works
        # for sentence in generator:  # works
        #     print(self.sentence_logprob(sentence))  # works
        ############################

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = {}  # might want to use defaultdict or Counter instead
        self.bigramcounts = {}
        self.trigramcounts = {}

        # Your code here
        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)

            for unigram in unigrams:
                self.unigramcounts[unigram] = self.unigramcounts.get(
                    unigram, 0) + 1

            for bigram in bigrams:
                self.bigramcounts[bigram] = self.bigramcounts.get(
                    bigram, 0) + 1

            for trigram in trigrams:
                self.trigramcounts[trigram] = self.trigramcounts.get(
                    trigram, 0) + 1
        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        numerator = self.trigramcounts.get(trigram, 0)
        denominator = self.bigramcounts.get(
            tuple(trigram[:2]), 0)
        if denominator == 0:
            return self.raw_unigram_probability((trigram[2],))
        return float(numerator/denominator)

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        numerator = self.bigramcounts.get(bigram, 0)
        denominator = self.unigramcounts.get((bigram[0],), 0)
        if denominator == 0:
            return self.raw_unigram_probability((bigram[0],))
        return float(numerator/denominator)

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        if unigram == ("START",):
            return 0.0
            # hint: recomputing the denominator every time the method is called
            # can be slow! You might want to compute the total number of words once,
            # store in the TrigramModel instance, and then re-use it.
        numerator = float(self.unigramcounts.get(unigram, 1))
        denominator = self.total_words
        return float(numerator/denominator)

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        return float(lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(trigram[1:]) + lambda3*self.raw_unigram_probability((trigram[2],)))

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        result = 0.0
        for trigram in trigrams:
            value = self.smoothed_trigram_probability(trigram)
            result += math.log2(value) if value > 0 else 0
        return result

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        log_sum = 0.0
        num_tokens = 0
        for sentence in corpus:
            num_tokens += len(sentence)
            log_sum += self.sentence_logprob(sentence)
        exponent = -1*float(log_sum/num_tokens)
        return 2**exponent


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp = model1.perplexity(corpus_reader(
            os.path.join(testdir1, f), model1.lexicon))
        # ..
        pp_model2 = model2.perplexity(corpus_reader(
            os.path.join(testdir1, f), model2.lexicon))
        total += 1
        correct += 1 if pp < pp_model2 else 0

    for f in os.listdir(testdir2):
        pp = model2.perplexity(corpus_reader(
            os.path.join(testdir2, f), model2.lexicon))
        # ..
        pp_model1 = model1.perplexity(corpus_reader(
            os.path.join(testdir2, f), model1.lexicon))
        total += 1
        correct += 1 if pp < pp_model1 else 0

    return float(correct/total)


if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # Part 1 Testing

    # print(get_ngrams(["natural", "language", "processing"], 1))  # works
    # print(get_ngrams(["natural", "language", "processing"], 2))  # works
    # print(get_ngrams(["natural", "language", "processing"], 3))  # works
    ############################

    # Part 2 Testing

    # print(model.trigramcounts[('START', 'START', 'the')])  # works
    # print(model.bigramcounts[('START', 'the')])  # works
    # print(model.unigramcounts[('the',)])  # works
    ############################

    # Part 3 Testing
    # for unigram in model.unigramcounts:  # works
    #     print(model.raw_unigram_probability(unigram))  # works
    # for bigram in model.bigramcounts:  # works
    #     print(model.raw_bigram_probability(bigram))  # works
    # for trigram in model.trigramcounts:  # works
    #     print(model.raw_trigram_probability(trigram))  # works
    ############################

    # Part 4 Testing

    # for trigram in model.trigramcounts:  # works
    #     print(model.smoothed_trigram_probability(trigram))  # works
    ############################

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[1], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)

    # Essay scoring experiment:
    # acc = essay_scoring_experiment(
    #     'train_high.txt', "train_low.txt", "test_high", "test_low")
    # print(acc)
