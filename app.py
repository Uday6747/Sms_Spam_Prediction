# pip install nltk scikit-learn

import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load SMS dataset
def load_data(file_path):
    messages = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label, message = line.strip().split('\t')
            messages.append(message)
            labels.append(label)
    return messages, labels

# Text preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove punctuation and numbers
    table = str.maketrans('', '', string.punctuation + string.digits)
    stripped = [token.translate(table) for token in tokens]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in stripped if token not in stop_words]
    
    # Join back to form a sentence
    processed_text = ' '.join(filtered_tokens)
    return processed_text

# Load and preprocess data
sms_data, sms_labels = load_data('SMS')
sms_data = [preprocess_text(message) for message in sms_data]

# Convert labels to binary (spam = 1, ham = 0)
sms_labels = [1 if label == 'spam' else 0 for label in sms_labels]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sms_data, sms_labels, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)
filename = 'cropmodel.pkl'
pickle.dump(classifier, open(filename, 'wb'))

# Predict on the test set
y_pred = classifier.predict(X_test_tfidf)

    
# Predict
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)
