#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# Load and clean the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Simple normalization
df['text'] = df['text'].str.lower()


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Convert text to numeric features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# In[3]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)


# Guide Questions:
# 1. What was the accuracy of your model?
# 
# - Most Accuracy of the model is 0.98% from the ham, that this dataset contains strong words like "free", "ham" and "price".
# 
# 2. Which class had higher recall: spam or ham? What does that mean?
# 
# - The (non-spam) ham had most of the recall and is higher which is 99% compared to the spam 92.95% means the model was slightly better at correctly identifying legitimate messages than catching all spam.
# 
# 3. Why do we lowercase the text before applying the model?
# 
# - Without lowercasing, the model might split its learning across case-sensitive duplicates, reducing efficiency and potentially harming performance. Normalization simplifies the feature space while preserving meaningful pattern
# 
# 4. What did you learn about how machine learning can classify raw text?
# 
# - This activity demonstrats that even minimally preprocessed text can yield strong classification results when distinctive word patterns exist. With the use of Naive Bayes model, despite its simplicity, effectively identified spam by leveraging word frequencies without needing advanced NLP techniques (like stemming or deep learning).
# 
# 5. What might you improve in the model or data if given more time?
# 
# - By manipulating most of the dataset by filtering out irrelivant words like "and", "the" to reduce noice. and removing punctuations also, in short by enhancing preprocessing if given more time to improve the model.
# 

# In[ ]:




