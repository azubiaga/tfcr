import gensim
import numpy as np
import re
import math
from os.path import join
import sys
from bert_embedding import BertEmbedding
import operator
from pprint import pprint

def loadW2vModel(w2vmodel):
    global model
    global index2word_set
    global num_features

    try:
        model
    except:
        if w2vmodel == 'google':
            data = 'GoogleNews-vectors-negative300.bin'
        elif w2vmodel == 'twitter':
            data = 'word2vec_twitter_model.bin'
        elif w2vmodel == 'wiki':
            data = 'glove.6B.300d.w2vformat.txt'
        elif w2vmodel == 'commoncrawl':
            data = 'glove.840B.300d.w2vformat.txt'

        print('Loading \'' + data + '\' w2v model...')

        if w2vmodel == 'twitter' or w2vmodel == 'google':
            model = gensim.models.KeyedVectors.load_word2vec_format(join('w2v-models', data), binary=True, unicode_errors='ignore')
        elif w2vmodel == 'commoncrawl' or w2vmodel == 'wiki':
            model = gensim.models.KeyedVectors.load_word2vec_format(join('w2v-models', data), binary=False, unicode_errors='ignore')

        index2word_set = set(model.wv.index2word)
        num_features = model.wv.syn0.shape[1]

        print('done!')

def tokenize(text):
    return re.sub(r'([^\s\w]|_)+', '', text.lower()).split()

def getw2vfeatures(text, getsum = 0):
    global model
    global index2word_set
    global num_features

    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.

    for word in tokenize(text):
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    if nwords > 1.0 and getsum == 0:
        featureVec = np.divide(featureVec,nwords)

    return np.asarray(featureVec)

def gettfcrw2vfeatures(text, labels, dataset, trainsize, getsum = 0, scatratio = {}, sfreq = {}):
    global model
    global index2word_set
    global num_features
    global catratio
    global freq

    catratio = scatratio
    freq = sfreq

    featureVec = []
    for key, label in enumerate(labels):
        category = dataset + '-' + trainsize + '-' + label
        featVec = np.zeros((num_features,),dtype="float32")
        totalcatr = 0.

        for word in tokenize(text):
            if word in catratio and label in catratio[word] and word in freq and label in freq[word]:
                catrword = catratio[word][label] * math.log(freq[word][label] + 1.0)
            elif word in catratio and label in catratio[word]:
                catrword = 1.0
            else:
                catrword = 0.0
            totalcatr += catrword

            if word + '@@' + category in index2word_set:
                featVec = np.add(featVec, model[word + '@@' + category] * catrword)
            elif word in index2word_set:
                featVec = np.add(featVec, model[word] * catrword)

        if totalcatr > 0.0 and getsum == 0:
            featVec = np.divide(featVec, totalcatr)

        featureVec.append(featVec)

    return np.concatenate(featureVec)

def gettfidfw2vfeatures(text, labels, dataset, trainsize, getsum = 0, sidf = {}, sfreq = {}):
    global model
    global index2word_set
    global num_features
    global idf
    global freq

    idf = sidf
    freq = sfreq

    featureVec = []
    for key, label in enumerate(labels):
        category = dataset + '-' + trainsize + '-' + label
        featVec = np.zeros((num_features,),dtype="float32")
        totalcatr = 0.

        for word in tokenize(text):
            if word in idf and label in idf[word] and word in freq and label in freq[word]:
                catrword = idf[word][label] * math.log(freq[word][label] + 1.0)
            elif word in idf and label in idf[word]:
                catrword = 1.0
            else:
                catrword = 0.0
            totalcatr += catrword

            if word + '@@' + category in index2word_set:
                featVec = np.add(featVec, model[word + '@@' + category] * catrword)
            elif word in index2word_set:
                featVec = np.add(featVec, model[word] * catrword)

        if totalcatr > 0.0 and getsum == 0:
            featVec = np.divide(featVec, totalcatr)

        featureVec.append(featVec)

    return np.concatenate(featureVec)

def getkldw2vfeatures(text, labels, dataset, trainsize, getsum = 0, skld = {}):
    global model
    global index2word_set
    global num_features
    global kld

    kld = skld

    featureVec = []
    for key, label in enumerate(labels):
        category = dataset + '-' + trainsize + '-' + label
        featVec = np.zeros((num_features,),dtype="float32")
        totalcatr = 0.

        for word in tokenize(text):
            if word in kld and label in kld[word]:
                catrword = kld[word][label]
            else:
                catrword = 0.0
            totalcatr += catrword

            if word + '@@' + category in index2word_set:
                featVec = np.add(featVec, model[word + '@@' + category] * catrword)
            elif word in index2word_set:
                featVec = np.add(featVec, model[word] * catrword)

        if totalcatr > 0.0 and getsum == 0:
            featVec = np.divide(featVec, totalcatr)

        featureVec.append(featVec)

    return np.concatenate(featureVec)

def getvectors(texts, dataset, features, trainsize, labels, sfreq = {}, scatratio = {}, sidf = {}, skld = {}):
    if features == 'gw2v' or features == 'gtfcrw2v' or features == 'gtfidfw2v' or features == 'gkldw2v':
        loadW2vModel(dataset, trainsize, distinct=0, largew2v=0, generic='google')
    if features == 'tw2v' or features == 'ttfcrw2v' or features == 'ttfidfw2v' or features == 'tkldw2v':
        loadW2vModel(dataset, trainsize, distinct=0, largew2v=0, generic='twitter')
    if features == 'cw2v' or features == 'ctfcrw2v' or features == 'ctfidfw2v' or features == 'ckldw2v':
        loadW2vModel(dataset, trainsize, distinct=0, largew2v=0, generic='commoncrawl')
    if features == 'ww2v' or features == 'wtfcrw2v' or features == 'wtfidfw2v' or features == 'wkldw2v':
        loadW2vModel(dataset, trainsize, distinct=0, largew2v=0, generic='wiki')

    vectors = []
    textcount = 0
    for text in texts:
        textcount += 1

        if features == 'gw2v' or features == 'tw2v' or features == 'cw2v' or features == 'ww2v':
            vectors.append(getw2vfeatures(text))
        if features == 'gtfidfw2v' or features == 'ttfidfw2v' or features == 'ctfidfw2v' or features == 'wtfidfw2v':
            vectors.append(gettfidfw2vfeatures(text, labels, dataset, trainsize, sidf=sidf, sfreq=sfreq))
        if features == 'gkldw2v' or features == 'tkldw2v' or features == 'ckldw2v' or features == 'wkldw2v':
            vectors.append(getkldw2vfeatures(text, labels, dataset, trainsize, skld=skld))
        if features == 'gtfcrw2v' or features == 'ttfcrw2v' or features == 'ctfcrw2v' or features == 'wtfcrw2v':
            vectors.append(gettfcrw2vfeatures(text, labels, dataset, trainsize, scatratio=scatratio, sfreq=sfreq))

    return vectors
