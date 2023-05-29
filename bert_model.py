
print('running')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

# import df1
df1 = pd.read_csv('./gpt_haikus1.csv', delimiter="$", header=None, names=["haiku", "AI"])
print(len(df1))

df2 = pd.read_csv('./lines.csv', delimiter="$", header=None, names=["haiku", "AI"])
print(len(df2))

combined_df = pd.concat([df1, df2])

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

batch_1 = combined_df

tokenized = batch_1["haiku"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()

labels = batch_1["AI"]

train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

lr_clf.score(test_features, test_labels)
print('finished')

