# %% [code]
# %% [code]
# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
#encoding:utf-8
import re
import nltk   #importing nlp library
import string   ##library string library that contains punctuation
string.punctuation

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer  #import Stemming function from nltk
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

# %% [code]
replacement = {
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't" : "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "i'd" : "I would",
    "i'll" : "I will",
    "i'm" : "I am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll":"it will",
    "i've" : "I have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll":" will",
    "tryin'":"trying",
}

# %% [code]
class NLPPreprocessor(object):
    def __init__(self,min_len = 2,stopwords_path = None):
        self.min_len = min_len
        self.stopwords_path = stopwords_path
        self.reset()

    def lower(self,sentence):
        '''
        Returns modified lower case text
        '''
        return sentence.lower()

    def reset(self):
        '''
        returns
        '''
        if self.stopwords_path:
            with open(self.stopwords_path,'r') as fr:
                self.stopwords = {}
                for line in fr:
                    word = line.strip(' ').strip('\n')
                    self.stopwords[word] = 1

    def remove_punctuation(self, text):
        #function to remove punctuation
        punctuationfree="".join([i for i in text if i not in string.punctuation])
        return punctuationfree
    
    def tokenization(self, text):
        #function for tokenization
        tokens = re.split('W+',text)
        return tokens
    
    def stemming(self, text):
        #defining a function for stemming    
        #defining the object for stemming
        porter_stemmer = PorterStemmer()
        stem_text = [porter_stemmer.stem(word) for word in text]
        return stem_text
 
    def lemmatizer(self, text):
        #function for lemmatization
        #defining the object for Lemmatization
        wordnet_lemmatizer = WordNetLemmatizer()       
        lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
        return lemm_text
    
    def remove_tabs(self, text):
        # function to remove the digits
        return re.sub('[^\w\s]',' ', text)

    def remove_digits(self, text):
        # function to remove the digits
        return re.sub('\d+', '', text)
    
    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    def remove_emoji(self, text):
        emoji_pattern = re.compile(
            '['
            u'\U0001F600-\U0001F64F'  # emoticons
            u'\U0001F300-\U0001F5FF'  # symbols & pictographs
            u'\U0001F680-\U0001F6FF'  # transport & map symbols
            u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
            u'\U00002702-\U000027B0'
            u'\U000024C2-\U0001F251'
            ']+',
            flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)


    def remove_html(self, text):
        html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        return re.sub(html, '', text)

    def remove_punct(self, text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)
    
    def clean_length(self,sentence):
        '''
        :param sentence:
        :return:
        '''
        if len([x for x in sentence]) >= self.min_len:
            return sentence

    def replace(self,sentence):
        '''
        :param sentence:
        :return:
        '''
        # Replace words like gooood to good
        sentence = re.sub(r'(\w)\1{2,}', r'\1\1', sentence)
        # Normalize common abbreviations
        words = sentence.split(' ')
        words = [replacement[word] if word in replacement else word for word in words]
        sentence_repl = " ".join(words)
        return sentence_repl

    def remove_website(self,sentence):
        '''
        :param sentence:
        :return:
        '''
        sentence_repl = sentence.replace(r"http\S+", "")
        sentence_repl = sentence_repl.replace(r"https\S+", "")
        sentence_repl = sentence_repl.replace(r"http", "")
        sentence_repl = sentence_repl.replace(r"https", "")
        return sentence_repl

    def remove_name_tag(self,sentence):
        # Remove name tag
        sentence_repl = sentence.replace(r"@\S+", "")
        return sentence_repl

    def remove_time(self,sentence):
        # Remove time related text
        sentence_repl = sentence.replace(r'\w{3}[+-][0-9]{1,2}\:[0-9]{2}\b', "")  # e.g. UTC+09:00
        sentence_repl = sentence_repl.replace(r'\d{1,2}\:\d{2}\:\d{2}', "")  # e.g. 18:09:01
        sentence_repl = sentence_repl.replace(r'\d{1,2}\:\d{2}', "")  # e.g. 18:09
        # Remove date related text
        # e.g. 11/12/19, 11-1-19, 1.12.19, 11/12/2019
        sentence_repl = sentence_repl.replace(r'\d{1,2}(?:\/|\-|\.)\d{1,2}(?:\/|\-|\.)\d{2,4}', "")
        # e.g. 11 dec, 2019   11 dec 2019   dec 11, 2019
        sentence_repl = sentence_repl.replace(
            r"([\d]{1,2}\s(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s[\d]{1,2})(\s|\,|\,\s|\s\,)[\d]{2,4}",
            "")
        # e.g. 11 december, 2019   11 december 2019   december 11, 2019
        sentence_repl = sentence_repl.replace(
            r"[\d]{1,2}\s(january|february|march|april|may|june|july|august|september|october|november|december)(\s|\,|\,\s|\s\,)[\d]{2,4}",
            "")
        return sentence_repl

    def remove_breaks(self,sentence):
        # Remove line breaks
        sentence_repl = sentence.replace("\r", "")
        sentence_repl = sentence_repl.replace("\n", "")
        sentence_repl = re.sub(r"\\n\n", ".", sentence_repl)
        return sentence_repl

    def remove_ip(self,sentence):
        # Remove phone number and IP address
        sentence_repl = sentence.replace(r'\d{8,}', "")
        sentence_repl = sentence_repl.replace(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', "")
        return sentence_repl

    def adjust_common(self,sentence):
        # Adjust common abbreviation
        sentence_repl = sentence.replace(r" you re ", " you are ")
        sentence_repl = sentence_repl.replace(r" we re ", " we are ")
        sentence_repl = sentence_repl.replace(r" they re ", " they are ")
        sentence_repl = sentence_repl.replace(r"@", "at")
        return sentence_repl

    def remove_chinese(self,sentence):
        # Chinese bad word
        sentence_repl = re.sub(r"fucksex", "fuck sex", sentence)
        sentence_repl = re.sub(r"f u c k", "fuck", sentence_repl)
        sentence_repl = re.sub(r"幹", "fuck", sentence_repl)
        return sentence_repl

    def full2half(self,sentence):
        '''
        :param sentence:
        :return:
        '''
        ret_str = ''
        for i in sentence:
            if ord(i) >= 33 + 65248 and ord(i) <= 126 + 65248:
                ret_str += chr(ord(i) - 65248)
            else:
                ret_str += i
        return ret_str

    def remove_stopwords(self,sentence):
        #function to remove stop words from the text and returns modified text
        stopwords = nltk.corpus.stopwords.words('english')
        words = sentence.split()
        x = [word for word in words if word not in stopwords]
        return " ".join(x)
    
    def remove_words(self,sentence,delwords):
        #function to remove stop words from the text and returns modified text
        words = sentence.split()
        x = [word for word in words if word not in delwords]
        return " ".join(x)

    #def __call__(self, sentence):
        '''
        x = sentence
        #x = self.lower(x)
        x = self.replace(x)
        x = self.remove_website(x)
        x = self.remove_name_tag(x)
        x = self.remove_time(x)
        x = self.remove_breaks(x)
        x = self.remove_ip(x)
        x = self.adjust_common(x)
        x = self.remove_chinese(x)
        '''
        #return x
    