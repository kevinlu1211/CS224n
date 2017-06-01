from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "squad")
    glove_dir = os.path.join("download", "dwr")
    source_dir = os.path.join("data", "squad")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    parser.add_argument("--random_init", default=True, type=bool)
    return parser.parse_args()


def basic_tokenizer(sentence):
    ## TODO: Use Stanford CoreNLP
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    output_words =[]
    for w in word:
        if w:
            output_words.extend(w)
        else:
            print("This word {} is not added to the sentence".format(w))
            print(w)
    return [w for w in words if w]


def initialize_vocabulary(vocabulary_path):
    # Returns two things: 
    # 1. A list of vocabs
    # 2. A dict of vocab where the dict has structure: {<word> : <index in list>}
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)]) # returns {<word> : <index in list>}
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def process_glove(args, vocab_list, save_path, size=4e5, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        if random_init:
            glove = np.random.randn(len(vocab_list), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_list), args.glove_dim))
        found = 0
        with open(glove_path, 'r') as fh:
            # Example of a line:
            # the -0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 -0.39141 0.3344 -0.57545 0.087459 0.28787 
            # -0.06731 0.30906 -0.26384 -0.13231 -0.20757 0.33395 -0.33848 -0.31743 -0.48336 0.1464 -0.37304 0.34577 
            # 0.052041 0.44946 -0.46971 0.02628 -0.54155 -0.15518 -0.14107 -0.039722 0.28277 0.14393 0.23464 -0.31021 
            # 0.086173 0.20397 0.52624 0.17164 -0.082378 -0.71787 -0.41531 0.20335 -0.12763 0.41367 0.55187 0.57908 
            # -0.33477 -0.36559 -0.54857 -0.062892 0.26584 0.30205 0.99775 -0.80481 -3.0243 0.01254 -0.36942 2.2167 
            # 0.72201 -0.24978 0.92136 0.034514 0.46745 1.1079 -0.19358 -0.074575 0.23353 -0.052062 -0.22044 0.057162 
            # -0.15806 -0.30798 -0.41625 0.37972 0.15006 -0.53212 -0.2055 -1.2526 0.071624 0.70565 0.49744 -0.42063 0.26148 
            # -1.538 -0.30223 -0.073438 -0.28312 0.37104 -0.25217 0.016215 -0.017099 -0.38984 0.87424 -0.72569 -0.51058 -0.52028 
            # -0.1459 0.8278 0.27062
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))

                # Iterate through all the words, and if it exists in the vocab list, then save the embedding at the index of the word
                # meaning that it will have the same index as the dictionary, so to access the word I just need to know the index of
                # the word which I can get through the {<word> : <index in list>} dictionary so something like word_matrix[lookup_dict[<word>]] 
                # to get the vector
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    # This function extracts the words from the files in the data_paths and creates a sorted list of vocabs used with one word on each line
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file: # data_path is the train.context, train.question etc files
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


if __name__ == '__main__':
    args = setup_args()
    vocab_path = pjoin(args.vocab_dir, "vocab.dat")

    train_path = pjoin(args.source_dir, "train")
    valid_path = pjoin(args.source_dir, "val")
    dev_path = pjoin(args.source_dir, "dev")

    # Creates a vocab.dat that has a word in every new time
    create_vocabulary(vocab_path,
                      [pjoin(args.source_dir, "train.context"),
                       pjoin(args.source_dir, "train.question"),
                       pjoin(args.source_dir, "val.context"),
                       pjoin(args.source_dir, "val.question")])

    # Creates a lookup dictionary and a list of words whose index is the same as the lookup dict
    vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below

    # Create a file which is the word matrix indexed by the lookup dict so we can do word_matrix[lookup_dict[<word>]] for the dwr
    process_glove(args, rev_vocab, args.source_dir + "/glove.trimmed.{}".format(args.glove_dim),
                  random_init=args.random_init)

    # ======== Creating Dataset =========
    # We created our data files seperately
    # If your model loads data differently (like in bulk)
    # You should change the below code
    
    # Creates a file of the sentence represented by their index in the lookup array
    x_train_ids_path = train_path + ".ids.context"
    y_train_ids_path = train_path + ".ids.question"
    data_to_token_ids(train_path + ".context", x_train_ids_path, vocab_path)
    data_to_token_ids(train_path + ".question", y_train_ids_path, vocab_path)

    x_ids_path = valid_path + ".ids.context"
    y_ids_path = valid_path + ".ids.question"
    data_to_token_ids(valid_path + ".context", x_ids_path, vocab_path)
    data_to_token_ids(valid_path + ".question", y_ids_path, vocab_path)