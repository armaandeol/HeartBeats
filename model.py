
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re


import os

# Specify the directory you want to walk through
directory_to_walk = "."  # This refers to the current directory

for dirname, _, filenames in os.walk(directory_to_walk):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import json

with open('intents.json' ,'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])


dic = {"tag": [], "patterns": [], "responses": []}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)

df = pd.DataFrame.from_dict(dic)
df['tag'].unique()

df_responses = df.explode('responses')
all_patterns = ' '.join(df['patterns'])


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


wordcloud = WordCloud(background_color='white', max_words=100, contour_width=3, contour_color='steelblue').generate(all_patterns)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Patterns')
plt.show()

df['pattern_length'] = df['patterns'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df['pattern_length'], bins=30, kde=True)
plt.title('Distribution of Pattern Lengths')
plt.xlabel('Length of Patterns')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(22, 16))
sns.countplot(y='tag', data=df, order=df['tag'].value_counts().index)
plt.title('Distribution of Intents')
plt.xlabel('Number of Patterns')
plt.ylabel('Intent')
plt.show()

df_unique_responses = df_responses.groupby('tag')['responses'].nunique().reset_index(name='unique_responses')
plt.figure(figsize=(22, 16))
sns.barplot(x='unique_responses', y='tag', data=df_unique_responses.sort_values('unique_responses', ascending=False))
plt.title('Number of Unique Responses per Intent')
plt.xlabel('Number of Unique Responses')
plt.ylabel('Intent')
plt.show()

df_responses['response_length'] = df_responses['responses'].apply(len)
plt.figure(figsize=(12, 8))
sns.histplot(df_responses['response_length'], bins=30, kde=True)
plt.title('Distribution of Response Lengths')
plt.xlabel('Length of Responses')
plt.ylabel('Frequency')
plt.show()


def preprocess_text(s):
    s = re.sub('[^a-zA-Z\']', ' ', s)  # Keep only alphabets and apostrophes
    s = s.lower()  # Convert to lowercase
    s = s.split()  # Split into words
    s = " ".join(s)  # Rejoin words to ensure clean spacing
    return s

# Apply preprocessing to the patterns
df['patterns'] = df['patterns'].apply(preprocess_text)
df['tag'] = df['tag'].apply(preprocess_text)

df['tag'].unique()

len(df['tag'].unique())
X = df['patterns']
y = df['tag']

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder

# Tokenization and Encoding the Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128  # Max sequence length


def encode_texts(texts, max_len):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


# Encoding labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_labels = len(np.unique(y_encoded))

# Encode the patterns
input_ids, attention_masks = encode_texts(X, max_len)
labels = torch.tensor(y_encoded)

# Splitting the dataset into training and validation
dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
validation_dataloader = DataLoader(val_dataset, batch_size=16)

# Model and Optimization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.2f}")


def predict_intent(text):
    # Tokenize and encode the text for BERT
    encoded_dict = tokenizer.encode_plus(
        text,  # Input text
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,  # Pad or truncate to max length
        pad_to_max_length=True,  # Pad to max length
        return_attention_mask=True,  # Construct attn. masks
        return_tensors='pt',  # Return pytorch tensors
    )

    # Extract input IDs and attention masks from the encoded representation
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    # No gradient calculation needed
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Use softmax to calculate probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

    # Get the predicted label with the highest probability
    predicted_label_idx = np.argmax(probabilities, axis=1).flatten()

    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform(predicted_label_idx)[0]

    return predicted_label, probabilities[0][predicted_label_idx]


# Example usage
test_message = "I feel anxious today"
predicted_intent = predict_intent(test_message)
print(f"Predicted Intent: {predicted_intent[0]}")

with open('intents.json', 'r') as file:
    intents_dict = json.load(file)
import random
tag_to_find = predicted_intent[0]

# Finding the index of the tag

# Finding the index of the tag
tag_index = next((index for index, intent in enumerate(intents_dict['intents']) if intent['tag'] == tag_to_find), None)

# if tag_index is not None:
#     print(f"The index of the tag '{tag_to_find}' is: {tag_index}")
# else:
#     print(f"The tag '{tag_to_find}' was not found.")
responses = intents_dict['intents'][tag_index]['responses'][random.randint(0, len(intents_dict['intents'][tag_index]['responses']) - 1)]
print(responses)

# Load intents only once to avoid reading the file multiple times
with open('intents.json', 'r') as file:
    intents_dict = json.load(file)


def get_response_from_intent(text):
    # Tokenize and encode the text for BERT
    encoded_dict = tokenizer.encode_plus(
        text,  # Input text
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_len,  # Pad or truncate to max length
        pad_to_max_length=True,  # Pad to max length
        return_attention_mask=True,  # Construct attention masks
        return_tensors='pt',  # Return pytorch tensors
    )

    # Extract input IDs and attention masks from the encoded representation
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    # No gradient calculation needed
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Use softmax to calculate probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

    # Get the predicted label with the highest probability
    predicted_label_idx = np.argmax(probabilities, axis=1).flatten()

    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform(predicted_label_idx)[0]

    # Finding the index of the tag
    tag_index = next(
        (index for index, intent in enumerate(intents_dict['intents']) if intent['tag'] == predicted_label), None)

    if tag_index is not None:
        # Randomly select a response from the corresponding responses
        response = random.choice(intents_dict['intents'][tag_index]['responses'])
    else:
        response = "I'm sorry, I don't understand."

    return predicted_label, response


# Example usage
test_message = "I am feeling a little bit happy"
predicted_intent, response = get_response_from_intent(test_message)
print(f"Predicted Intent: {predicted_intent}")
print(f"Response: {response}")

test_message = input("Describe How you are feeling today?")
predicted_intent, response = get_response_from_intent(test_message)
print(f"Response: {response}")
