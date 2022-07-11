import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from DeepCTR.deepctr.models.autofuse import AUTOFUSE
from DeepCTR.deepctr.layers.core import PredictionLayer, DNN
from DeepCTR.deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import time


if __name__ == "__main__":
    with open('dataset/CDs_new.pkl', 'rb') as f:
        data = pickle.load(f)

    dense_coarse_features = ['price']
    dense_fine_features = []

    sparse_user_features = ['reviewerID','reviewerName']
    sparse_coarse_features =  ['vote', 'unixReviewTime','brand','style']
    sparse_fine_features = ['asin','reviewText','title', 'summary','imageURL','description']
    pop_features = ['popularity']
    label_features = ['label']

    sparse_features = sparse_user_features + sparse_coarse_features + sparse_fine_features
    dense_features = dense_coarse_features + dense_fine_features
    data[sparse_features] = data[sparse_features].fillna('-1', ) 
    data[dense_features] = data[dense_features].fillna(0, ) 
    target = label_features

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features + pop_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    item_num = data['asin'].nunique()

    # 2.count #unique features for each sparse field,and record dense feature field name
    coarse_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim = 4)
                           for i,feat in enumerate(sparse_coarse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_coarse_features]
    fine_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim = 4)
                           for i,feat in enumerate(sparse_fine_features)] 
    user_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim = 4)
                           for i,feat in enumerate(sparse_user_features)]
    pop_feature_columns = [DenseFeat(feat,1,) for feat in pop_features]
    label_feature_columns = [DenseFeat(feat,1,) for feat in label_features]

    feature_names = get_feature_names(fine_feature_columns + coarse_feature_columns + user_feature_columns + pop_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2, random_state=2018)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
  
    # 4. build the model
    start = time.time()
    model = AUTOFUSE(user_feature_columns,coarse_feature_columns,fine_feature_columns, pop_feature_columns, label_feature_columns, pop_vocab_size=item_num) 
    model.compile("adam", ['binary_crossentropy']*3,metrics= ['AUC'])
    history = model.fit(train_model_input, [train[target].values]*3,
                    batch_size=512, epochs=3, verbose=1, validation_split=0.2,)

    print("/------------------AutoFuse------------------/")
    pred_ans = model.predict(test_model_input, batch_size=512)
    CTR_logloss = round(log_loss(test[target].values, pred_ans[0]), 4)
    CTR_AUC = round(roc_auc_score(test[target].values, pred_ans[0]), 4)
    end = time.time()

    print("CTR: test LogLoss", CTR_logloss)
    print("CTR: test AUC", CTR_AUC)
    print("time", end - start)

