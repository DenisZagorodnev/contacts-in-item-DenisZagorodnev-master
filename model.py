from typing import Tuple, Union
import pandas as pd
import re
import string
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from pymorphy2 import MorphAnalyzer
from natasha import NamesExtractor, MorphVocab, DatesExtractor, MoneyExtractor, AddrExtractor
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import scipy as sc


def get_title(title):
    
    return re.split('-| ', title)[:3]

def get_date(data):
    
    data['year'] = data['datetime_submitted'].dt.year
    
    data['month'] = data['datetime_submitted'].dt.month
    
    return data

def encode_feature(data, feature):
    
    data = np.asarray([[item] for item in list(data[feature])])
    
    encoder = OrdinalEncoder()

    result = encoder.fit_transform(data)
    
    return result

def round_prices(price):
    
    return np.round(price / 100)*100

morph_vocab = MorphVocab()

def sub_names(line):
    line = str(line)
    
    extractor = NamesExtractor(morph_vocab)
    
    matches = extractor(line)
    
    if len([_.fact.as_json for _ in matches]) == 3:
        return   1
      
    else:
        return 0



def sub_addr(line):
    
    extractor = AddrExtractor(morph_vocab)
    
    matches = extractor(line)
    
    if len([_.fact.as_json for _ in matches]):
        return  1
    else: 
        return 0

def sub_phone_number(line):
    line = line.translate(str.maketrans('', '', string.punctuation))
    res = re.findall(r'[7-8]?\s?[0-9][0-9][0-9]?\s?[0-9][0-9][0-9]?\s?[0-9][0-9]?\s?[0-9][0-9]$', line)
    
    if (len(res) > 0) or ('телефон' in line) or ('звонить' in line):
        return 1
    else:
        return 0

def len_description(line):
    
    return len(line.split())

morph = MorphAnalyzer()

def preproc_line(line):
  
    line = ''.join(i for i in line if not i.isdigit())
    
    line = line.translate(str.maketrans('', '', string.punctuation))
    
    line = line.lower()
    
    line = [morph.parse(word)[0].normal_form for word in line]
    
    line = ' '.join(line)

    return line


def preproc_data(train_data, content):
    
    corpus = [''.join(preproc_line(line)) for line in list(train_data[content].astype(str))]

    return corpus




def preproc_data_new(data):
    
    data['datetime_submitted'] = pd.to_datetime(data['datetime_submitted'], errors='coerce')
    
    data['number_founded'] = data['description'].apply(lambda x: sub_phone_number(x))
    
    data['year'] = data['datetime_submitted'].dt.year
    
    data['month'] = data['datetime_submitted'].dt.month
    
    data['len_description'] = data['description'].apply(lambda x: len_description(x))
    
    data['price'] = data['price'].apply(lambda price: round_prices(price))
    
    for feature in ['subcategory', 'category', 'region', 'city']:
        
        data[feature + '_encoded']  = encode_feature(data, feature)
        
    return data


def preproc_line_5(line):
  
    line = ''.join(i for i in line if not i.isdigit())
    
    line = line.translate(str.maketrans('', '', string.punctuation))
    
    line = line.lower()

    return line


def preproc_data_5(train_data, content):
    
    corpus = [''.join(preproc_line_5(line)) for line in list(train_data[content].astype(str))]

    return corpus



def task1(data_train_origin, data_test):
    
    data_train = pd.DataFrame(columns = list(data_train_origin.columns))
    
    cats = list(set(data_train_origin['category']))

    frames = []
    
    for cat in cats:
        
        df_sub = data_train_origin[data_train_origin['category'] == cat]
        
        df_sub = df_sub.sample(n = 100000, replace=True)
        
        frames.append(df_sub)
        
    data_train = pd.concat(frames)
    
    
    data_train_proceed = preproc_data_5(data_train,'description')

    data_test_proceed = preproc_data_5(data_test,'description')
    
    vectorizer = TfidfVectorizer(ngram_range = (1, 2))
    
    X_train_vectorized = vectorizer.fit_transform(data_train_proceed)

    X_test_vectorized = vectorizer.transform(data_test_proceed)
    
    data_train_proceed = preproc_data_new(data_train)

    data_test_proceed = preproc_data_new(data_test)
    
    y_train = data_train_proceed['is_bad']
    X_train = data_train_proceed[[ 'price', 'number_founded','month', 'subcategory_encoded', 'category_encoded',
           'region_encoded', 'city_encoded', 'len_description']]

    X_test = data_test_proceed[[ 'price', 'number_founded', 'month', 'subcategory_encoded', 'category_encoded',
           'region_encoded', 'city_encoded', 'len_description']]
    
    
    X_train_scr = sc.sparse.csr_matrix(X_train.values)
    X_train_new = sc.sparse.hstack((X_train_vectorized, X_train_scr))
    
    X_test_scr = sc.sparse.csr_matrix(X_test.values)
    X_test_new = sc.sparse.hstack((X_test_vectorized, X_test_scr))
    
    model_6 = xgb.XGBRegressor(objective ='rank:pairwise',  
                         base_score=0.4,
                         colsample_bytree = 0.4, 
                         learning_rate = 0.1,
                max_depth = 55, alpha = 45, n_estimators = 45)

    model_6 = model_6.fit(X_train_new, y_train)
    
    preds = model_6.predict(X_test_new)

    #return len(title) % 3 / 2
    
    return preds





def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
