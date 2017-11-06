# -*- coding: utf-8 -*-

import os
import pandas as pd
import nltk
import csv
import re

##os.chdir(r'C:\Users\ritesh\Desktop\TextMining\Project Files')
#path=r'C:\Users\ritesh\Desktop\TextMining\Project Files' 
#mode=777
#os.chmod(path, mode)

file = 'osha.xlsx'
datafile = pd.read_excel(file,header=None,names=["ID","cause","summary","summary1","summary2"])
summary = datafile["summary"]
#test_data = datafile


def activities(datafile):
    activities_final_list=[]
    for index,row in datafile.iterrows():
        text=row[2]
        cleaned_text=re.findall(r'(?<=was).*?(?=\.){1}',text)
        cleaned_text=cleaned_text[:1]
        for text in cleaned_text:
            text_token=nltk.word_tokenize(text)
            text_pos=nltk.pos_tag(text_token)
            answer=[]
            pattern = "VERB: {<VBG><IN>?<TO>?<VB>?<RP>?<JJ>?<DT>?(<NN>|<NNS>)+}"
            NPchunk = nltk.RegexpParser(pattern)
            result = NPchunk.parse(text_pos)
            print(result)
            for subtree in result.subtrees(filter=lambda t: t.label() == 'VERB'):
                for i in subtree.leaves():
                    answer.append(i[0])
                    result_last = ' '.join(answer)
                activities_final_list.append(result_last)
    return activities_final_list    


activities_list=activities(datafile)
activities_list=[[x] for x in activities_list]
with open("Activities.csv","w") as f:
        wr=csv.writer(f,quoting=csv.QUOTE_ALL)
        wr.writerow(['Activities'])
        wr.writerows(activities_list)

