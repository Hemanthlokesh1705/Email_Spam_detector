import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,classification_report
url=r"machine Leaning\spam.csv"#use your relative path 
df=pd.read_csv(url)
print(df.head())
df['Category']=df['Category'].map({'spam':1,'ham':0})
print(df["Category"].head())
y=df['Category']
x=df['Message']
vector=CountVectorizer(stop_words='english')
x_vector=vector.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x_vector, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
prec=precision_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Presicion: {prec:.2f}")
print(classification_report(y_test, y_pred))
plt.scatter(y_test,y_pred)
new_email = ["Congratulations! You have won a free lottery. Click here to claim your prize."]
new_email_vectorized = vector.transform(new_email)
ans=model.predict(new_email_vectorized)
if ans[0]==1:
    print("spam")
else:
    print("ham")
#Descion boundry
from sklearn.datasets import make_classification


# Generate synthetic dataset (2 features for easy visualization)
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_redundant=0, random_state=42)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create a meshgrid for plotting decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))

# Predict on the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary of Logistic Regression")
plt.show()
