# Copyright 2016, Google, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START sentiment_tutorial]
"""Demonstrates how to make a simple call to the Natural Language API."""

import matplotlib
matplotlib.use('Agg')
# [START sentiment_tutorial_import]
import argparse
import csv
import numpy as np

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
# [END sentiment_tutorial_import]

import matplotlib.pyplot as plt

# [START def_print_result]
def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('Sentence {} has a sentiment score of {}'.format(
            index, sentence_sentiment))

    print('Overall Sentiment: score of {} with magnitude of {}'.format(
        score, magnitude))
    return 0
# [END def_print_result]


# [START def_analyze]
def analyze(movie_review_filename):
    """Run a sentiment analysis request on text within a passed filename."""
    client = language.LanguageServiceClient()

    with open(movie_review_filename, 'r') as review_file:
        # Instantiates a plain text document.
        content = review_file.read()
#        content.encode('utf-8').strip()
#        content = unicode(content, "utf-8")
    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude
    L=len(annotations.sentences)
    if L > 0:
      return [score, magnitude/L]
    else:
      return [score, magnitude]
    # Print the results
#    print_result(annotations)
# [END def_analyze]


if __name__ == '__main__':
#    parser = argparse.ArgumentParser(
#        description=__doc__,
#        formatter_class=argparse.RawDescriptionHelpFormatter)
#    parser.add_argument(
#        'movie_review_filename',
#        help='The filename of the movie review you\'d like to analyze.')
#    args = parser.parse_args()

#    arrays=["articles/test.txt","articles/test3.txt","articles/test2.txt"]
#    for X in arrays:
#        analyze(X)

    articles_path = "Articles"
    X=np.empty([0,2])

    count=0
    lastval=-10000
    lastresult=0
    flag=0
    mydict={}
    with open('EventArticles.csv') as csvfile:
        readCSV=csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            sentiment_result=analyze(articles_path+"/"+row[1])
            #vprint row[1]
            #vprint sentiment_result[0]
            if flag<>0 and lastval<>row[0]:
                lastresult=lastresult/count
                count=0
                mydict[lastval]=lastresult
                row_new=[lastval,lastresult]
                X=np.vstack([X,row_new])
                #vprint row_new
                lastresult=0
            flag=1
            lastval=row[0]
            count=count+1
            lastresult=lastresult+sentiment_result[0]
        
        lastresult=lastresult/count
        count=0
        mydict[lastval]=lastresult
        row_new=[lastval,lastresult]
        X=np.vstack([X,row_new])
        #vprint row_new
        lastresult=0

    with open('Scores.csv','wb') as csvWriteFile:
        writeCSV=csv.writer(csvWriteFile,delimiter=",")
        writeCSV.writerows(X)


    Day0=np.empty([0,2])
    Day1=np.empty([0,2])
    Day2=np.empty([0,2])
    with open('Crimelist.csv') as csvfile:
        readCSV=csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            score = mydict[row[0]]
            crimerate = float(row[1])*100
            row_new=[score,crimerate]
#            if row[2]=='0':
            Day0=np.vstack([Day0,row_new])
#            if row[2]=='1':
#                Day1=np.vstack([Day1,row_new])
#            if row[2]=='2':
#                Day2=np.vstack([Day2,row_new])
    print (Day0)
#    print (Day1)   
    with open('Day0.csv','wb') as csvWriteFile:
        writeCSV=csv.writer(csvWriteFile,delimiter=",")
        writeCSV.writerows(Day0)
    
