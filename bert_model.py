
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

a = lr_clf.score(test_features, test_labels)
print(f'model score: {a}')

# testing out our model on some haikus
haiku1 = "teardrops stain pages / heartbreak's ink on love's story / healing begins now"
haiku2 = "White sands on the beach / And pink petals off a branch / Drifting in the wind"

new_input_ids1 = torch.tensor(tokenizer.encode(haiku1, add_special_tokens=True)).unsqueeze(0)
new_outputs1 = model(new_input_ids1)
new_last_hidden_states1 = [new_outputs1[0].detach().numpy()[0][0]]
haiku1_predictions = lr_clf.predict_proba(new_last_hidden_states1)

new_input_ids2 = torch.tensor(tokenizer.encode(haiku2, add_special_tokens=True)).unsqueeze(0)
new_outputs2 = model(new_input_ids2)
new_last_hidden_states2 = [new_outputs2[0].detach().numpy()[0][0]]
haiku2_predictions = lr_clf.predict_proba(new_last_hidden_states2)

print(f'Haiku 1: {haiku1}\nLikelihood of being written by ChatGPT: {haiku1_predictions[0]}\nLikelihood of being written by a human: {haiku1_predictions[1]}')
print(f'Haiku 2: {haiku2}\nLikelihood of being written by ChatGPT: {haiku2_predictions[0]}\nLikelihood of being written by a human: {haiku2_predictions[1]}')

print('finished')