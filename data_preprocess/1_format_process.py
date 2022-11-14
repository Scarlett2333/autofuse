import pickle
import pandas as pd
import re
def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      line = re.sub("\"verified\"\: true,",'',line)
      line = re.sub("\"verified\"\: false,",'',line)
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df


reviews_df = to_df('ft_local/CDs_and_Vinyl_5.json')

with open('dataset/reviews_CDs.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('ft_local/meta_CDs_and_Vinyl.json')
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open('dataset/meta_CDs.pkl', 'wb') as f:
  pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)


