# -*- coding: utf-8 -*-
"""
@author: Ritesh
"""

import nltk
from nltk import word_tokenize, pos_tag

# ===== POS(PART OF SPEECH) Tagging using NLTK =====

sent = '''Professor Tan Eng Chye, NUS Deputy President and Provost, and Professor 
Menahem Ben-Sasson, President of HUJ signed the joint degree agreement at NUS, 
in the presence of Ambassador of Israel to Singapore Her Excellency Amira Arnon 
and about 30 invited guests, on July 03, 2013.
'''

# The input for POS tagger needs to be tokenized first.
sent_pos = pos_tag(word_tokenize(sent))
sent_pos

# A more simplified tagset - universal
sent_pos2 = pos_tag(word_tokenize(sent), tagset='universal')
sent_pos2

# The wordnet lemmatizer works properly with the pos given
wnl = nltk.WordNetLemmatizer()
wnl.lemmatize('signed', pos = 'v')

#------------------------------------------------------------------------
# Exercise: remember the wordcloud we created last week? Now try creating 
# a wordcloud with only nouns, verbs, adjectives, and adverbs, with nouns 
# and verbs lemmatized.
#-------------------------------------------------------------------------

noun_are = [t[0] for t in sent_pos2 if t[1] == 'NOUN']
noun_are

adjective_are = [a[0] for a in sent_pos2 if a[1] == 'ADJ']
adjective_are

adverb_are = [b[0] for b in sent_pos2 if b[1] == 'ADV']
adverb_are

verb_are = [v[0] for v in sent_pos2 if v[1] == 'VERB']
verb_are

lem_noun = [wnl.lemmatize(d) for d in noun_are]
lem_noun

lem_verb = [wnl.lemmatize(e) for e in verb_are]
lem_verb

text_clean=" ".join(noun_are)
text_clean

import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

wc = WordCloud(background_color="white").generate(text_clean)

# Display the generated image:
# the matplotlib way:

plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

wc.to_file("example.png")
