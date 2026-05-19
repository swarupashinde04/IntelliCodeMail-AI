import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("emails.csv")

# Inputs and outputs
X = data["text"]

y = data["label"]

# Convert text into numbers
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Custom test
message = ["Win free money now"]

message_vector = vectorizer.transform(message)

result = model.predict(message_vector)

print("Prediction:", result[0])