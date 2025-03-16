import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.datasets import make_classification

data_path = r"machine Leaning\spam.csv"
df = pd.read_csv(data_path)

df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})

labels = df['Category']
messages = df['Message']

vectorizer = CountVectorizer(stop_words='english')
message_vectors = vectorizer.fit_transform(messages)

X_train, X_test, y_train, y_test = train_test_split(message_vectors, labels, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(classification_report(y_test, predictions))

plt.scatter(y_test, predictions)

test_email = ["Congratulations! You have won a free lottery. Click here to claim your prize."]
test_email_vectorized = vectorizer.transform(test_email)
result = model.predict(test_email_vectorized)

if result[0] == 1:
    print("spam")
else:
    print("ham")

features, labels = make_classification(n_samples=200, n_features=2, n_classes=2, n_redundant=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

decision_model = LogisticRegression()
decision_model.fit(X_train, y_train)

x_range, y_range = np.meshgrid(np.linspace(features[:, 0].min()-1, features[:, 0].max()+1, 100),
                               np.linspace(features[:, 1].min()-1, features[:, 1].max()+1, 100))

boundary_predictions = decision_model.predict(np.c_[x_range.ravel(), y_range.ravel()])
boundary_predictions = boundary_predictions.reshape(x_range.shape)

plt.figure(figsize=(8, 6))
plt.contourf(x_range, y_range, boundary_predictions, alpha=0.3, cmap='coolwarm')
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary of Logistic Regression")
plt.show()
