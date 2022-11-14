import random
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import math
# from numba import jit

random.seed(1234)

with open('dataset/reviews_CDs.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  reviews_df = reviews_df[['overall','reviewerID','reviewerName', 'asin','vote','reviewText', 'unixReviewTime','summary','style']]
  reviews_df['style'] = reviews_df['style'].map(lambda x: x['Format:'] if type(x) != float else '-1')

  # print(len(reviews_df))


with open('dataset/meta_CDs.pkl', 'rb') as f:
  meta_df = pickle.load(f)
  meta_df = meta_df[['asin','imageURL','description','title','price','brand']]
  meta_df['imageURL'] = meta_df['imageURL'].map(lambda x: x[0] if x != [] else '-1')
  meta_df['description'] = meta_df['description'].map(lambda x: x[0] if x != [] else '-1')
  meta_df['price'] = meta_df['price'].map(lambda x: float(x[1:].replace(',','')) if x!="" and x[0]=='$' else 0.0)  # 有些数据存在混乱乱码
  

data = pd.merge(reviews_df, meta_df)
order = ['overall', 'asin','vote','price','reviewerID','reviewerName','reviewText', 'unixReviewTime','summary','imageURL','description', 'title','brand','style']
data = data[order]
# convert overall score to label(<=3:0; >3:1)
data.overall[data.overall<=3] = 0
data.overall[data.overall>3] = 1


print(data)

# 统计pop
count = pd.DataFrame(data['asin'].value_counts())
print(type(count))
count = count.reset_index()

count.columns=['asin','pop']
print(count)
data = pd.merge(data,count)



# split old ads and new ads
count = pd.DataFrame(data['asin'].value_counts())
count.columns=['nums']
old = count[count['nums'] >= 20]
delindexs = old.index
pd.set_option('display.max_columns', None)
old_ads = data[data['asin'].isin(delindexs)]
new_ads = data[~data['asin'].isin(delindexs)]

print(len(old_ads))
print(len(new_ads))


with open('dataset/CDs_old.pkl', 'wb') as f:
  pickle.dump(old_ads, f, pickle.HIGHEST_PROTOCOL)

with open('dataset/CDs_new.pkl', 'wb') as f:
  pickle.dump(new_ads, f, pickle.HIGHEST_PROTOCOL)
