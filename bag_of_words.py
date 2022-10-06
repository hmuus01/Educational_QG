from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from os import path

import pandas as pd
from matplotlib import pyplot as plt
w = path.join('results_sciqleaf2.csv')

# unknown
unknown_words = pd.read_csv(w)
unknowns = []
for unks in unknown_words['words'].to_numpy():
    if len(unks):
        unknowns.extend(unks)

#uknowns=['another blah', '大元', '大都', 'ł', 'ł', '大元通制', 'ἄζωτον', 'κτείς']

abstracts = path.join('data','science_full','cs.csv')

# SCIQ
df_abstracts = pd.read_csv(abstracts)

vectorizer = CountVectorizer()
cv_fit = vectorizer.fit_transform(df_abstracts['text'])
data_names=vectorizer.get_feature_names()

unknowns = list(set(unknowns))
for unk in unknowns:
    if unk not in data_names:
        print(f'Word: {unk} | not in dataset')
    else:
        print(f'Word: {unk} | in dataset')


# print(data_names)
freqs = np.asarray(cv_fit.sum(axis=0))[0] #cv_fit.toarray().sum(axis=0)
print(freqs)
freqs_sorted = np.argsort(-freqs)
print(freqs_sorted)

names = np.array(data_names)[freqs_sorted]
scores = freqs[freqs_sorted][:20]
names = names[:20]
# plt.bar(names, scores)
# plt.show()

# print(names[:20])

