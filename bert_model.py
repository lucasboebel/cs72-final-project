import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import torch
import transformers as ppb
import warnings

# Fine-Tuning a BERT for AI- and Human-Generated Poetry Classification
# Ben Barris (benjamin.f.barris.25@dartmouth.edu),
# Lucas Boebel (lucas.j.boebel.23@dartmouth.edu),
# Jacob Donoghue (jacob.l.donoghue.22@dartmouth.edu)
# Last modification: 2023/06/03

# Scaffolding from:
# Rolando Coto-Solano (Rolando.A.Coto.Solano@dartmouth.edu)
# Dartmouth College, LING48/CS72, Spring 2023

warnings.filterwarnings("ignore")

# import df1
df1 = pd.read_csv(
    "./gpt4_haikus2.csv", delimiter="$", header=None, names=["haiku", "AI"]
)

# import df2
df2 = pd.read_csv("./lines.csv", delimiter="$", header=None, names=["haiku", "AI"])

# combine the datasets, making sure an equal number of human- and AI-generated poems are in the training set
batch_1 = pd.concat([df1, df2[: len(df1)]])

model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel,
    ppb.DistilBertTokenizer,
    "distilbert-base-uncased",
)
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = batch_1["haiku"].apply(
    (lambda x: tokenizer.encode(x, add_special_tokens=True))
)

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:, 0, :].numpy()

labels = batch_1["AI"]

# separate training and testing features
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels
)

# train logistic regression model
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

a = lr_clf.score(test_features, test_labels)
print(f"model score: {a}")

# evaluate model
predictions = lr_clf.predict(test_features)
print(classification_report(test_labels, predictions))

cf_matrix = confusion_matrix(test_labels, predictions)
print("confusion matrix", cf_matrix)
print('------')

# testing out our model on some haikus not in the dataset to test efficacy visually 
print('Testing sample haikus with 0.5 threshold')
strings = [
    "whispering breeze blows / petals dance upon the wind / fragrance fills the air",
    "overhead knocking / headboards joyfully mocking / louder loneliness",
    "playing through the night / music echoes through the soul / the same song repeats"
]
labels = [1, 0, 0]
for i in range(len(strings)):
    new_input_ids = torch.tensor(
        tokenizer.encode(strings[i], add_special_tokens=True)
    ).unsqueeze(0)
    new_outputs = model(new_input_ids)
    new_last_hidden_states = [new_outputs[0].detach().numpy()[0][0]]
    predictions = lr_clf.predict_proba(new_last_hidden_states)
    prediction = 0
    if predictions[0][0] < 0.5:
        prediction = 1
    print("haiku:", strings[i])
    print("predicted label", prediction)
    print("prediction confidence:", predictions[0][prediction])
    print("actual label", labels[i])
    print("-----")

print('----------- calculated to minimize false positives \n ------------ \n')
# run the model again, this time with a threshold of 0.9973052, to minimize false positives
test_probabilities = lr_clf.predict_proba(test_features)
test_predictions2 = []
max_fp_probability = 0
test_labels_arr = list(test_labels) # convert so we can index into it

for i in range(len(test_probabilities)):
    prediction = 0
    probability_ai = test_probabilities[i][1]
    if (probability_ai >= 0.9973052):
        prediction = 1
    
    # identify maximum false positive confidence
    test_predictions2.append(prediction)
    if (test_labels_arr[i] == 0 and prediction == 1):
        # false positive
        if (probability_ai > max_fp_probability):
            max_fp_probability = probability_ai
print('maximum false positive probability', max_fp_probability)
print(classification_report(test_labels, test_predictions2))

cf_matrix = confusion_matrix(test_labels, test_predictions2)
print('confusion matrix:', cf_matrix)
print('----')

print('--------Testing sample haikus with 0.9958324 threshold----------')

for i in range(len(strings)):
    new_input_ids = torch.tensor(
        tokenizer.encode(strings[i], add_special_tokens=True)
    ).unsqueeze(0)
    new_outputs = model(new_input_ids)
    new_last_hidden_states = [new_outputs[0].detach().numpy()[0][0]]
    predictions = lr_clf.predict_proba(new_last_hidden_states)
    prediction = 0
    if predictions[0][1] >= 0.9958324:
        prediction = 1
    print("haiku:", strings[i])
    print("predicted label", prediction)
    print("prediction confidence:", predictions[0][prediction])
    print("actual label", labels[i])
    print("-----")