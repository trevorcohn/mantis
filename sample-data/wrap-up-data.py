import sys
import collections
import itertools

def threshold_vocab(fname, threshold):
    word_counts = collections.Counter()
    with open(fname) as fin:
        for line in fin:
            for token in line.split():
                word_counts[token] += 1

    ok = set()
    for word, count in sorted(word_counts.items()):
        if count >= threshold:
            ok.add(word)
    return ok

def process_corpus(sf, tf, of, sv=source_vocab, tv=target_vocab):
    with open(of, 'w') as fout:
        with open(sf) as sin:
            with open(tf) as tin:
                for sline, tline in itertools.izip(sin, tin):
                    print >>fout, '<s>',
                    for token in tline.split():
                        if token in target_vocab:
                            print >>fout, token,
                        else:
                            print >>fout, '<unk>',
                    print >>fout, '</s>', '|||',

		    print >>fout, '<s>',
                    for token in sline.split():
                        if token in source_vocab:
                            print >>fout, token,
                        else:
                            print >>fout, '<unk>',
                    print >>fout, '</s>'

def process_test(sf, of, sv=source_vocab):
    with open(of, 'w') as fout:
        with open(sf) as sin:
                for sline in sin:
                    print >>fout, '<s>',
                    for token in sline.split():
                        if token in source_vocab:
                            print >>fout, token,
                        else:
                            print >>fout, '<unk>',
                    print >>fout, '</s>'


sfname = 'train.de'
tfname = 'train.en'

source_vocab = threshold_vocab(sfname, 5) #5 is word frequency threshold
target_vocab = threshold_vocab(tfname, 5)

process_corpus(sfname, tfname, 'train.de-en.unk.cap', source_vocab, target_vocab) #train
process_corpus('dev.de', 'dev.en', 'dev.de-en.unk.cap', source_vocab, target_vocab) #dev
process_corpus('test.de', 'test.en', 'test.de-en.unk.cap', source_vocab, target_vocab) #test

