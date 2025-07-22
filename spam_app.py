import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack, csr_matrix # Import hstack and csr_matrix
import numpy as np # Import numpy for array operations

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') # Explicitly download as it was requested in a previous error traceback


# --- Load Trained Models and Scaler ---
@st.cache_resource
def load_models():
    """Loads the TF-IDF Vectorizer, MinMaxScaler, and the trained Spam Detector model."""
    try:
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        spam_detector_model = joblib.load('spam_detector_model.pkl')
        scaler = joblib.load('scaler.pkl') # Load the scaler

        # --- Diagnostic: Check feature consistency ---
        # The model expects TF-IDF features + 4 engineered features
        vectorizer_n_features = len(tfidf_vectorizer.vocabulary_)
        expected_total_features = vectorizer_n_features + 4 # 4 for engineered features
        model_n_features = spam_detector_model.n_features_in_

        # st.info(f"Loaded TF-IDF Vectorizer has {vectorizer_n_features} features.")
        # st.info(f"Loaded Spam Detector Model expects {model_n_features} features (incl. 4 engineered features).")
        # st.info(f"Combined features (TF-IDF + Engineered) should be: {expected_total_features}")


        if model_n_features != expected_total_features:
            st.error(
                f"**Feature Mismatch Error (Diagnostic):** The loaded Spam Detector Model expects {model_n_features} features, "
                f"but the combination of TF-IDF ({vectorizer_n_features}) and engineered features (4) results in {expected_total_features} features. "
                "This indicates an inconsistency between your saved model files or the feature engineering process. "
                "Please ensure all components (vectorizer, model, scaler) were saved from the *same* training pipeline and are compatible."
            )
            return None, None, None # Return None for all if mismatch
        # --- End Diagnostic ---

        return tfidf_vectorizer, spam_detector_model, scaler
    except FileNotFoundError:
        st.error("Error: One or more model files (tfidf_vectorizer.pkl, spam_detector_model.pkl, or scaler.pkl) not found.")
        st.info("Please make sure all .pkl files are in the same directory as this Streamlit app.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.info("Please verify your .pkl files are not corrupted and are compatible with the current library versions.")
        return None, None, None


tfidf_vectorizer, spam_detector_model, scaler = load_models()

# --- Text Preprocessing Function (as used during training) ---
def preprocess_text(text):
    """
    Applies the same preprocessing steps used during model training.
    This includes lowercasing, removing punctuation, tokenization,
    stop word removal, and stemming.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stop words and apply stemming
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stopwords_set]
    return ' '.join(processed_tokens)

# --- Spam Prediction Function ---
def predict_spam(email_text, vectorizer, model, scaler):
    """
    Predicts if a given text is spam using the loaded TF-IDF vectorizer,
    MinMaxScaler, and the trained model, including engineered features.
    """
    if vectorizer is None or model is None or scaler is None:
        return "Error: Models or scaler not loaded."

    # 1. Clean and transform text using TF-IDF vectorizer
    cleaned_text = preprocess_text(email_text)
    tfidf_vec = vectorizer.transform([cleaned_text]) # Ensure it's a sparse matrix

    # 2. Extract and scale engineered features
    char_count = len(email_text)
    word_count = len(email_text.split())
    punctuation_count = sum([1 for c in email_text if c in string.punctuation])
    uppercase_word_count = sum(1 for w in email_text.split() if w.isupper())

    # Create a numpy array for engineered features and reshape for scaler
    meta_features = np.array([
        char_count,
        word_count,
        punctuation_count,
        uppercase_word_count
    ]).reshape(1, -1) # Reshape to (1, 4) for a single sample

    meta_scaled = scaler.transform(meta_features)

    # 3. Combine TF-IDF features and scaled engineered features
    # Ensure both are sparse matrices before hstack if possible, or convert meta_scaled to csr_matrix
    final_features = hstack([tfidf_vec, csr_matrix(meta_scaled)])

    # Make prediction
    prediction = model.predict(final_features)

    # Return "Spam" or "Not Spam" based on your model's output
    # Assuming 1 for spam, 0 for not spam based on common binary classification
    return "Spam" if prediction[0] == 1 else "Not Spam"

# --- Streamlit Web Page Layout ---

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("📧 Spam Detector")
st.markdown("""
    This application helps you detect whether a given text message is spam or not.
    Enter your message in the text area below and click 'Predict'.
""")

# Check if models and scaler were loaded successfully before proceeding
if tfidf_vectorizer is not None and spam_detector_model is not None and scaler is not None:
    # Text input area for the user
    user_input = st.text_area("Enter your message here:", height=150, placeholder="Type your message...")

    # Prediction button
    if st.button("Predict"):
        if user_input:
            # Get prediction from the actual model
            prediction = predict_spam(user_input, tfidf_vectorizer, spam_detector_model, scaler)

            # Display the result
            if prediction == "Spam":
                st.error(f"**Prediction:** This message is likely **{prediction}!**")
                st.balloons() # Add some fun for spam detection!
            elif prediction == "Not Spam":
                st.success(f"**Prediction:** This message is **{prediction}.**")
            else:
                st.warning(prediction) # Display error message from predict_spam
        else:
            st.warning("Please enter some text to get a prediction.")
else:
    st.warning("Model loading failed. Please check the console for errors and ensure model files are present and compatible.")


# --- Instructions to run ---
# To run this app:
# 1. Save the code as a Python file (e.g., `spam_app.py`).
# 2. Make sure 'tfidf_vectorizer.pkl', 'spam_detector_model.pkl', and 'scaler.pkl' are in the same directory.
# 3. Open your terminal or command prompt.
# 4. Navigate to the directory where you saved the file.
# 5. Run the command: `streamlit run spam_app.py`
# 6. Your web browser will automatically open the app.
