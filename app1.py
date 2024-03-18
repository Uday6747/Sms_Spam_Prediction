import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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

# Load the trained model
def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

# Predict function
def predict_spam(input_text, model, vectorizer):
    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)
    
    # Convert to TF-IDF vector
    tfidf_vector = vectorizer.transform([preprocessed_text])
    
    # Predict using the model
    prediction = model.predict(tfidf_vector)
    
    return "spam" if prediction[0] == 1 else "ham"

if __name__ == "__main__":
    # Load the trained model and vectorizer
    model = load_model('cropmodel.pkl')
    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(X_train)  # Make sure to use the same vectorizer used during training

    # Sample input text for prediction
    input_text = "Congratulations! You have won a free iPhone. Claim now!"
    
    # Predict
    result = predict_spam(input_text, model, vectorizer)
    print(f"Input text: {input_text}")
    print(f"Prediction: {result}")
