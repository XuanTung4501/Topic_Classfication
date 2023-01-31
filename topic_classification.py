import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("dataset.csv")

le = LabelEncoder()

le.fit(data['label'])

labels_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

data['label'] = le.transform(data['label'])

text = data['text'].values
topics = data['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text, topics, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LinearSVC())
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict the topics of the test data
predictions = pipeline.predict(X_test)

# Evaluate the performance of the classifier using f1 score
f1 = f1_score(y_test, predictions, average='macro')
print("Labels mapping: ", labels_mapping)
print("F1 Score:", f1)
