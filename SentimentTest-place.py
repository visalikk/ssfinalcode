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

# [START sentiment_tutorial_import]
import argparse
import csv
import numpy as np

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
# [END sentiment_tutorial_import]


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
    keyFile = 'Place_key_final.csv'
    articles_path = "Articles"
    X=np.empty([0,2])
    lastresult=0
    count=0
    lastval=-10000
    lastresult=0
    mydict={}
    
    with open("hypothetical_test1.txt", 'r') as the_file:
        key = the_file.read()
        print str(key)    
# Read location key file then filter out only i/p area process those history & predict. 
    with open(keyFile) as csvfile:
        readCSV=csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            if row[0]== key: 
               value = row[1] 
               print (row[1])
               
               with open('EventArticles.csv') as csvfile:
                 readCSV=csv.reader(csvfile,delimiter=',')
                 for row in readCSV:
                    if row[0] == value:
                      hypothetical_result=analyze(articles_path+"/"+row[1])
                      lastresult=lastresult+hypothetical_result[0] 
                      count=count+1
    lastresult=lastresult/count
    mydict[lastval]=lastresult
    row_new=[lastval,lastresult]
    X=np.vstack([X,row_new])
    with open('hypothetical_sentiment1.csv','wb') as csvWriteFile:
      writeCSV=csv.writer(csvWriteFile,delimiter=',')
      print hypothetical_result
      writeCSV.writerow(X)

# [END sentiment_tutorial]
