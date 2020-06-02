import sys
from os import listdir, mkdir
from os.path import isfile, join, exists
import re

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier

import math

import featuregeneration
import evaluation

if ' ' in sys.argv[1]:
    chunks = sys.argv[1].split()
    dataset = chunks[0]
    features = chunks[1]
    trainsize = chunks[2]
else:
    dataset = sys.argv[1]
    features = sys.argv[2]
    trainsize = sys.argv[3]

print(dataset + ' ' + features + ' ' + trainsize)

classifs = ['maxent', 'randomforest', 'nb', 'svm', 'mlp', 'gp', 'gbc']
if len(sys.argv) > 4:
    classifs = [sys.argv[4]]
classifiers = []

done = 0
for classifier in classifs:
    outfile = classifier + '_' + dataset + '_' + features + '_' + trainsize

    if exists(join('results', outfile)):
        done += 1
    else:
        classifiers.append(classifier)

if done == len(classifs):
    print('Done earlier!')
    sys.exit()

id_preds = {}
id_gts = {}
for classifier in classifiers:
    id_preds[classifier] = {}
    id_gts[classifier] = {}

diffgts = []
foldgts = []
foldwordcounts = []
for fold in range(0, 10):
    foldwordcount = []
    foldgt = []
    with open(join('datasets', dataset, 'data-' + str(fold)), 'r', encoding='utf-8') as fh:
        for line in fh:
            data = line.split('\t')
            gt = data[1]
            text = ''
            if len(data) > 2:
                text = data[2].strip()
            text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
            tokens = list(set(re.sub(r'([^\s\w]|_)+', '', text.strip().lower()).split(' ')))
            foldwordcount.append(tokens)
            foldgt.append(gt)
            if not gt in diffgts:
                diffgts.append(gt)

    foldwordcounts.append(foldwordcount)
    foldgts.append(foldgt)

for fold in range(0, 10):
    model = {}
    foldids = []
    foldvectors = []

    sfreq = {}
    scatratio = {}
    sidf = {}
    skld = {}
    incgts = {}
    doccount = 0
    for fold2 in range(0, 10):
        if fold2 != fold:
            for doc, categ in zip(foldwordcounts[fold2], foldgts[fold2]):
                doccount += 1
                if doccount <= int(trainsize):
                    incgts[categ] = 1 + incgts.get(categ, 0)
                    for word in doc:
                        if not word in sfreq:
                            sfreq[word] = {}
                        sfreq[word][categ] = 1 + sfreq[word].get(categ, 0)

    for word, wordcounts in sfreq.items():
        scatratio[word] = {}
        sidf[word] = {}
        skld[word] = {}
        total = sum(wordcounts.values())
        for categ, wordcount in wordcounts.items():
            scatratio[word][categ] = float(wordcount) / float(total)
            sidf[word][categ] = math.log(float(incgts[categ])) / float(wordcount)

            p = float(wordcount) / float(incgts[categ])
            q = float(sum(wordcounts.values())) / float(sum(incgts.values()))
            skld[word][categ] = max(p * math.log(p / q), 0.0)

    traincount = 0
    for fold2 in range(0, 10):
        elcount = 0
        foldid = []
        foldtext = []
        with open(join('datasets', dataset, 'data-' + str(fold2)), 'r', encoding='utf-8') as fh:
            for line in fh:
                data = line.strip().split('\t')
                foldid.append(data[0])
                if len(data) > 2:
                    apptext = data[2]
                else:
                    apptext = ''
                foldtext.append(apptext)

        if int(trainsize) > 10000 or fold == fold2 or (int(trainsize) <= 10000 and fold2 < 2): # memory saving
            foldvector = featuregeneration.getvectors(foldtext, dataset, features, trainsize, diffgts, sfreq = sfreq, scatratio = scatratio, sidf = sidf, skld = skld)

        foldids.append(foldid)
        foldvectors.append(foldvector)

    X_train = []
    y_train = []
    traincount = 0
    for fold2 in range(0, 10):
        if fold2 != fold:
            for vector, gt in zip(foldvectors[fold2], foldgts[fold2]):
                traincount += 1
                if traincount <= int(trainsize):
                    X_train.append(vector)
                    y_train.append(gt)

    X_test = foldvectors[fold]
    y_test = foldgts[fold]

    for classifier in classifiers:
        if not exists('folds/' + classifier + '_' + dataset + '_' + features + '_' + trainsize + '-' + str(fold)):
            if classifier == "svm":
                model[classifier] = LinearSVC(class_weight='balanced')
            if classifier == "nb":
                model[classifier] = GaussianNB()
            if classifier == "randomforest":
                model[classifier] = RandomForestClassifier(n_estimators=10, class_weight='balanced')
            if classifier == "maxent":
                model[classifier] = LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced')
            if classifier == 'mlp':
                model[classifier] = MLPClassifier()
            if classifier == 'gp':
                model[classifier] = GaussianProcessClassifier(optimizer=None, max_iter_predict=25, multi_class='one_vs_one')
            if classifier == 'gbc':
                model[classifier] = GradientBoostingClassifier()

            model[classifier].fit(X_train, y_train)

            y_pred = model[classifier].predict(X_test)
        else:
            y_pred = []
            with open('folds/' + classifier + '_' + dataset + '_' + features + '_' + trainsize + '-' + str(fold), 'r') as fh:
                for line in fh:
                    datav = line.strip().split('\t')
                    y_pred.append(datav[1])

        acc = 0
        items = 0
        for k, itemid in enumerate(foldids[fold]):
            id_preds[classifier][itemid] = y_pred[k]
            id_gts[classifier][itemid] = y_test[k]

            items += 1
            if y_pred[k] == y_test[k]:
                acc += 1

        if items > 0:
            outfile = classifier + '_' + dataset + '_' + features + '_' + trainsize
            print(outfile + ' - ' + str(fold) + ': ' + str(float(acc) / items))

for classifier in classifiers:
    outfile = classifier + '_' + dataset + '_' + features + '_' + trainsize

    with open(join('predictions', outfile), 'w') as fw:
        for tweetid, gt in id_gts[classifier].items():
            fw.write(str(tweetid) + '\t' + str(id_preds[classifier][tweetid]) + '\t' + str(gt) + '\n')

    evaluation.evaluate(id_preds[classifier], id_gts[classifier], outfile)
