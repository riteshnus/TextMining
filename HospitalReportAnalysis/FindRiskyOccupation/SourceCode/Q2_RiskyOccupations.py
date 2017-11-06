# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 01:40:27 2017

@author: Ritesh
"""

import nltk
import os
import re
import json
import unicodedata
import string
import io
import pandas as pd
import sklearn
import numpy as np
global collections
import collections
global operator
import operator
global create_tag_image
global make_tags
global LAYOUTS
global get_tag_counts
from nltk import *
from nltk.corpus import stopwords
from nltk import word_tokenize

#Using Panda to read the preprocessed file
df = pd.ExcelFile('C:\\D\\NUS\\Sem4\\TextMining\\CA\\PreprocessedData\\filtered_construction_data_refined.xlsx').parse('filtered_construction_data_refi')
#df.columns = ['id', 'title', 'summary']


list=df.ix[0:,2]
print (list[50])
occupationList=[]
#Reading each paragraph to fetch occupation
for i in list[0:]:
    occupationList1 = []
#normalize text
    ntext=unicodedata.normalize('NFKD', i).encode('ascii','ignore')

#convert to string to apply transalation
    finalText = str(ntext)
#removed puntuation marks
    withoutPunctuation = finalText.translate(finalText.maketrans("",""))
#lowering all text
    textForAnalysis=withoutPunctuation.lower()
#removing all english stop words
    stop = stopwords.words('english')
#New stop words added
    additional_stop_list = ['employer','container','water','boiler','barrier','trimmer','outdoor','hammer','elevator','helicopter','power',
    'reactor','motor','escavator','chamber','paper','blower','door','shoulder','fiber','bulldozer','cover','computer','giver',
    'cutter','scissor','tanker','closer','member','man','upper','steer','subfloor','interior','conveyor','number','rubber','bumper',
    'master','tractor','polyester','boulder','lever','floor','compressor','21inchdiameter','preheater','indoor','30indiameter','refrigerator','carrier','detonator','ladder',
    'ladder','exterior','tower','bladder','tower','stater','crusher','slicer','starter','senior','mower','baler','auger','powder','odor','other','cinder','sensor','mixer',
    'vapor','galvanometer','blaster','trucktractor','tractortrailer','receiver','chipper','boilermaker','dishwasher','default','order','man','operator' ,
    'truck','either','trimmer','center','meter','threeman','diameter','hopper','finger','number','heater','manufacturer','condenser','roller','respirator','customer','generator','stepladder','examiner','december','november','october','september',
    'transformer','zipper','leftover','hyster','collector','cylinder','laser','enter','proper','freezer','outer','monitor','super','manner',
    'andor','river','pusher','lighter','remember','twoman','inner','smaller','color','liver','layer','calender','sorter','liver','pioneer','danger','evaporator','riser',
    'recover','indicator','pier','server','radiator','upriver','transfer','factor','crawler','copper','waether','counter','separator','shower','coker','workover',
    'alligator','perimeter','cooler','ranger','offcenter','poor','dockdoor','door','waterpower','manipulator','together','hoter','bdoor','feeler','autoleather',
    'containertailer','dinner','voltmeter','behavior','partner','winter','summer','sucker','sandpaper','thinner','chiller','miter','remainder','regulator',
    'thirdfloor','horsepower','rotor','sprinkler','corner','weather','secondfloor','plaster','trigger','tier','longer','minor','gutter','outrigger','choker','coffer','primer','adapter','screwdriver','er','median','brother','spider',
    'extinguisher','capacitor','answer',
    'insulator','longer','vaibrator','bunker','hanger','escalator','however','corridor','lower','underwater','fastener','booster','trailer', 'compactor', 'dumpster', 'girder', 'spreader',
    'lacquer', 'werner', 'kaiser','uncover', 'solder', 'dumper', 'timber', 'oiler', 'sealer','whaler', 'wastewater', 'carburetor', 'vibrator', 'erector', 'picker','taper','sewerwater,'
    'marker', 'absorber','firstfloor', 'maneuver', 'chocker',  'lather','pedestrian', 'tensioner','exchanger','badger', 'precipitator', 'improper', 'ether',
    'rollercompactor','teeter','deadman', 'shaker', 'estimator','lasher', 'trencher', 'incinerator','cotter','connector', 'dozer','dormer','jackhammer','harbor','conditioner','skidsteer','stringer','rafter','liner','greater','mirror','sandblaster',
    'marker','sewerwater','header','beer','blocker','fever','rollover','equalizer','snooper','consider','connector','finisher','scraper','grader','anchor','lumber','grinder']

#removing more stop words using my list
    stop.extend(additional_stop_list)
    text_nostop=" ".join(filter(lambda word: word not in stop, textForAnalysis.split()))
#doing pos tagging
    pos1 = pos_tag(word_tokenize(text_nostop))
#taking only NN words to analysis Occupation
    text1= [word for word, pos in pos1 if (pos == 'NN')]
    print ("NN Words")
    print (text1)
# Fetching only words ending with er,or,ian,man,men
    regex1=re.compile(".*(er$)|.*(or$)|.*(ian$)|.*(anic$)|.*(man$)|.*(men$)")


    empolypee=['worker','supervisor','driver','laborer','contractor','foreman','owner','carpenter','technician','manager','welder','excavator','painter',
                   'conductor','sewer','lumber','engineer','journeyman','labor', 'electrician','subcontractor','helper','mechanic',
                   'driller','officer','washer','instructor','lineman','brakeman','painter', 'inspector','ironworker', 'cleaner','rider',
                   'plumber','flagman','fabricator','builder','repairman','caster','warehouseman','signalman','groundsman','pipefitter','motorman' ,'controller',
                   'presser','fireman','framer','landscaper','dealer','pressman','craftsman','derdger','technician','mechanic','fencer','operator','ironwroker',
                   'landscaper','decorator','pipefitter','fixer','steelfixer','welder','waterproofer','worker','sitemanager','Site Manager','material handler' ,
                   'engineer','architect','civilman','plasterer','pile driver','rigger' ]

#First, we check with exhaustive list made for construction occupation
    #pos1=["hello","VBOT","pressman"]
    for i in text1:
        is_break = False
        for a in empolypee:
            if(i==a):
                occupationList1=i
                print ("Break- Matched in Exact List Search ")
                print (occupationList1)
                occupationList.append(occupationList1)
                is_break = True
                break
        if is_break: break
# If occupation not found in list then using regular expression which used er,or,anic words to fetch as occupation
    print (occupationList1)
    if occupationList1==[]:
        print ("welcome")
        occupationList1=[m.group(0) for l in text1 for m in [regex1.search(l)] if m]
        print ("Words")
        print (occupationList1)
        if not occupationList1:
             occupationList1=["no_occupation"]

        occupationList.append(occupationList1[0])

#Output all fetched occupation in Output

dd = df.ix[0:,0:3]
dd['Occupation'] = pd.Series(occupationList, index=df.index)
writer=pd.ExcelWriter('output.xlsx')
dd.to_excel(writer)
writer.save()

print ("Final List")
print (occupationList)

#Print count of occupation
counts = collections.Counter(occupationList)
print (counts)

from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts
j=""
for i in occupationList:
    if(i!='no_occupation'):
        j=j+" "+i
print (j)

# Let's plot the result
import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

mask = np.array(Image.open("fly.png"))
image_colors = ImageColorGenerator(mask)

wc3 = WordCloud(background_color='white', mask=mask).generate(j)

plt.imshow(wc3.recolor(color_func=image_colors))
plt.axis("off")
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('fly_output.png', dpi=100)
plt.show()


