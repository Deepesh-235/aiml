import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv('emails.csv')  # Assumes 'text' and 'spam' columns

# 2. Clean text (optional for basic model)
df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True).str.lower()

# 3. Convert text to features
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

# 4. Set labels
y = df['spam']  # 0 = ham, 1 = spam

# 5. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 6. Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
