import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocess import clean_text

df = pd.read_csv("data/fake_job_postings.csv")

df['text'] = (
    df['title'].fillna('') + " " +
    df['company_profile'].fillna('') + " " +
    df['description'].fillna('') + " " +
    df['requirements'].fillna('') + " " +
    df['benefits'].fillna('')
)

df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['fraudulent']

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
