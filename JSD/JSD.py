import collections

import pandas as pd
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import itertools

from shifterator import shifts as rs
from shifterator import shifts as ss

stop_words = set(stopwords.words('english'))

# data preprocessing
def preprocess(text):
  '''Removes hashtags and converts links to [URL] and usernames starting with @ to [USER],
  it also converts emojis to their textual form.'''
  documents = []
  for instance in text:
    instance = re.sub(r'@([^ ]*)', '[USER]', instance)
    instance = re.sub(r'&amp', '[USER]', instance)

    instance = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '[URL]', instance)
    instance = re.sub(r'[USER]', '', instance)
    instance = re.sub(r'[URL]', '', instance)
    instance = instance.replace('#', '')
    documents.append(instance)
  return documents



def remove_punctuation(txt):
    """Replace URLs and other punctuation found in a text string with nothing
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with URLs and punctuation removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


def remove_stopwords(txt):
    """Removes punctuation, changes to lowercase, removes
        stopwords, and calculates word frequencies.

    Parameters
    ----------
    txt : string
        A text string that you want to clean.

    Returns
    -------
    Words and frequencies
    """

    tmp = [remove_punctuation(t) for t in txt]
    tmp = [t.lower().split() for t in tmp]

    tmp = [[w for w in t if not w in stop_words]
           for t in tmp]

    tmp = list(itertools.chain(*tmp))
    tmp = collections.Counter(tmp)

    return tmp


df_original = pd.read_csv("df_A.csv")
df_original = df_original[['Premise', 'Stance']]
df_original.replace({'Stance': {'against': 1, 'in favor of': 0}}, inplace=True)

df_additional = pd.read_csv("df_B.csv")
df_additional = df_additional[['Premise', 'Stance']]

clean_texts_original = preprocess(df_original['Premise'].tolist())
clean_texts_additional = preprocess(df_additional['Premise'].tolist())

clean_texts_original = remove_stopwords(clean_texts_original)
clean_texts_additional = remove_stopwords(clean_texts_additional)



jsd_shift = ss.JSDivergenceShift(type2freq_1=clean_texts_original,
                                type2freq_2=clean_texts_additional,
                                base=2,
                                alpha=1)
jsd_shift.get_shift_graph(system_names = ['Dataset A', 'Dataset B'],
                          title='JSD Shift of Dataset A and Dataset B')
print(jsd_shift.diff)