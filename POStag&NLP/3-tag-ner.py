# -*- coding: utf-8 -*-
"""
@author: Ritesh
"""

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# ===== POS Tagging and NER(NAMED ENTITY RECOGNITION) using NLTK =====

sent = '''Professor Tan Eng Chye, NUS Deputy President and Provost, and Professor 
Menahem Ben-Sasson, President of HUJ signed the joint degree agreement at NUS, 
in the presence of Ambassador of Israel to Singapore Her Excellency Amira Arnon 
and about 30 invited guests, on Sept 25, 2013.
'''

# The input for POS tagger needs to be tokenized first.
sent_pos = pos_tag(word_tokenize(sent))
sent_pos

# ===== NER using NLTK =====
# The input for the NE chunker needs to have POS tags.
sent_chunk = ne_chunk(sent_pos)
print(sent_chunk)

# ===== Now try creating your own named entity and noun phrase chunker ====
# We need to define the tag patterns to capture the target phrases and use 
# RegexParser to chunk the input with those patterns.
# Some minimal tag patterns are given here. 

#grammar = r"""
#  NE: {<NNP>+(<IN>)?(<NNP>)?}      # chunk sequences of proper nouns
#  NP:                 
#      {<DT><NN>}     
#"""

grammar = r"""
    NE: {<NNP>+(<IN|CC><NNP>)*(<TO><NNP>)*}
    NP: {<CD|DT><JJ>*<NN|NNS>+}
"""

cp = nltk.RegexpParser(grammar)
print(cp.parse(sent_pos))

#------------------------------------------------------------------------
# Exercise: modify the above tag patterns to capture the NEs and NPs in the 
# example sentence. 
#-------------------------------------------------------------------------