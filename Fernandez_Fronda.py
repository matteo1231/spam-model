import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# App title
st.title("📩 SMS Spam Classifier")
st.write("A simple NLP classifier using Naive Bayes to detect spam messages.")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['text'] = df['text'].str.lower()
    return df

df = load_data()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Show accuracy
accuracy = accuracy_score(y_test, y_pred)
st.metric(label="📊 Model Accuracy", value=f"{accuracy:.2%}")

# Input form
st.header("Try it yourself!")
user_input = st.text_area("Enter an SMS message to classify", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vector = vectorizer.transform([user_input.lower()])
        prediction = model.predict(input_vector)[0]
        st.success(f"The message is **{prediction.upper()}**.")

# Toggle for classification report
with st.expander("🔍 See model performance details"):
    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)

# Footer
st.markdown("---")
st.caption("Made by Fernandez & Fronda — Streamlit NLP Demo")
