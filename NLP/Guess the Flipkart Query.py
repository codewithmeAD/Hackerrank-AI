import urllib.request
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Function to clean and extract words
def words(text):
    return re.findall(r'(?:[a-zA-Z]+[a-zA-Z\'\-]?[a-zA-Z]|[a-zA-Z]+)', text)

### Step 1: Load training data
train_url = "https://hr-testcases.s3.amazonaws.com/2552/assets/training.txt"
x_train = []
y_train = []

response = urllib.request.urlopen(train_url)
lines = response.read().decode('utf-8').splitlines()

for idx, line in enumerate(lines):
    if idx == 0:  # skip header
        continue
    parts = line.rstrip().split("\t")
    if len(parts) != 2:
        continue
    text, label = parts
    sen = " ".join(word for word in words(text))
    x_train.append(sen)
    y_train.append(label)

x = np.array(x_train)
y = np.array(y_train)

### Step 2: Train classifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC())
])
text_clf.fit(x, y)

### Step 3: Load test data from URL
test_url = "https://hr-testcases.s3.amazonaws.com/2552/assets/sampleInput.txt"
test_response = urllib.request.urlopen(test_url)
test_lines = test_response.read().decode('utf-8').splitlines()

test_data = [" ".join(word for word in words(line)) for line in test_lines]

### Step 4: Predict and print
predicted = text_clf.predict(np.array(test_data))
for label in predicted:
    print(label)
