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
from copy import copy

# значимый сегмент названия

def get_title(title):
    
    return re.split('-| ', title)[:3]

#вынуть год и месяц из даты

def get_date(data):
    
    data['year'] = data['datetime_submitted'].dt.year
    
    data['month'] = data['datetime_submitted'].dt.month
    
    return data

#общий инструмент кодирования фичей

def encode_feature(data, feature):
    
    data = np.asarray([[item] for item in list(data[feature])])
    
    encoder = OrdinalEncoder()

    result = encoder.fit_transform(data)
    
    return result

#приведение цен

def round_prices(price):
    
    return np.round(price / 100)*100


morph_vocab = MorphVocab()

#вынуть имя

def sub_names(line):
    line = str(line)
    
    extractor = NamesExtractor(morph_vocab)
    
    matches = extractor(line)
    
    if len([_.fact.as_json for _ in matches]) > 0:
        return   1
      
    else:
        return 0

#вынуть телефонный номер

def sub_phone_number(line):
    line = line.translate(str.maketrans('', '', string.punctuation))
    res = re.findall(r'[7-8]?\s?[0-9][0-9][0-9]?\s?[0-9][0-9][0-9]?\s?[0-9][0-9]?\s?[0-9][0-9]$', line)
    
    if (len(res) > 0) or ('телефон' in line) or ('звонит' in line):
        return 1
    else:
        return 0

#длина описания

def len_description(line):
    
    return len(line.split())

#количество заглавных букв

def sub_upper(line):
    
    upper_count = sum(map(str.isupper, line))
    
    return upper_count

#вынуть электронную почту

def sub_gmail(line):
    
    res = re.findall(r'@gmail|@yandex', line)
    
    if len(res) > 0:
        return 1
    else:
        return 0
    
#удалить стоп-слова

def del_stopwords(line, words_pack):
    
    filtered_line = []
    
    for word in line.split(' '):
        
        if word not in  words_pack:
            
            filtered_line.append(word)
            
    return filtered_line


morph = MorphAnalyzer()

#обработка сэмпла описания

def preproc_line(line, stop_words):
    
    line = re.sub('\n|\r|\t', '', line)
  
    line = ''.join(i for i in line if not i.isdigit())
    
    line = line.translate(str.maketrans('', '', string.punctuation))
    
    line = line.lower()
    
    line = re.sub(' +', ' ', line)
    
    #line = [morph.parse(word)[0].normal_form for word in line]
    
    #line = ' '.join(line)
    
    if len(stop_words):
        
        line = del_stopwords(line, stop_words)
        
        line = ' '.join(line)

    return line

#обработка корпуса описаний

def preproc_data(train_data, content, stop_words = []):
    
    corpus = [''.join(preproc_line(line, stop_words)) for line in list(train_data[content].astype(str))]

    return corpus



#сгенерить все фичи и кодировать

def sub_description(data):
    
    data['upper_count'] = data['description'].apply(lambda x: sub_upper(x))
    
    data['datetime_submitted'] = pd.to_datetime(data['datetime_submitted'], errors='coerce')
    
    data['number_founded'] = data['description'].apply(lambda x: sub_phone_number(x))
    
    data['year'] = data['datetime_submitted'].dt.year
    
    data['month'] = data['datetime_submitted'].dt.month
    
    data['len_description'] = data['description'].apply(lambda x: len_description(x))
    
    data['price'] = data['price'].apply(lambda price: round_prices(price))
    
    data['count_digits'] = data['description'].apply(lambda x: sum(c.isdigit() for c in x))
    
    data['count_spec'] = data['description'].apply(lambda x: len(re.sub('[\w]+' ,'', x)))
    
    data['len_title'] = data['title'].apply(lambda x: len_description(x))
    
    data['mail_founded'] = data['description'].apply(lambda x: sub_gmail(x))
    
    for feature in ['subcategory', 'category', 'region', 'city']:
        
        data[feature + '_encoded']  = encode_feature(data, feature)
        
    return data




def task1(data_train_origin, data_test):
    
    data_train = copy(data_train_origin)
    
    #data_train = pd.DataFrame(columns = list(data_train_origin.columns))
    
    #cats = list(set(data_train_origin['category']))

    #frames = []
    
    #for cat in cats:
        
    #    df_sub = data_train_origin[data_train_origin['category'] == cat]
    #   
    #    df_sub = df_sub.sample(n = 100000, replace=True)
    #   
    #    frames.append(df_sub)
    #    
    #data_train = pd.concat(frames)
    
    
    data_train_proceed = sub_description(data_train)

    data_test_proceed = sub_description(data_test)
    
    
    data_train_desc_proceed = preproc_data(data_train_proceed, 'description', [])

    data_test_desc_proceed = preproc_data(data_test_proceed, 'description', [])
    
    vectorizer = TfidfVectorizer(ngram_range = (1, 2))

    X_train_vectorized = vectorizer.fit_transform(data_train_desc_proceed)

    X_test_vectorized = vectorizer.transform(data_test_desc_proceed)
    
    
    features = [
     'price',
     'upper_count',
     'number_founded',
     'year',
     'month',
     'len_description',
     'count_digits',
     'count_spec',
     'len_title',
     'mail_founded',
     'subcategory_encoded',
     'category_encoded',
     'region_encoded',
     'city_encoded']
    
    y_train = data_train_proceed['is_bad']
    X_train = data_train_proceed[features]
    
    y_test = data_test_proceed['is_bad']
    X_test = data_test_proceed[features]
    
    
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
