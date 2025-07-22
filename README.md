📧 Spam Detection Web App
This repository contains a Streamlit web application for real-time spam detection. It leverages a machine learning model trained to classify text messages (or short emails) as either "Spam" or "Not Spam" (Ham). The application provides an intuitive interface for users to input text and receive instant predictions.

✨ Features
Interactive Web Interface: Built with Streamlit for easy text input and prediction display.

Machine Learning Powered: Utilizes a Multinomial Naive Bayes classifier.

Advanced Text Preprocessing: Incorporates TF-IDF vectorization for text feature extraction.

Enriched Feature Set: Enhances prediction accuracy by including engineered features such as character count, word count, punctuation count, and uppercase word count.

Real-time Predictions: Get instant classification results as you type or paste text.

🚀 Technologies & Libraries Used
This project is built using Python and relies on the following key libraries:

Streamlit: For creating the interactive web application.

Scikit-learn (sklearn): For machine learning functionalities, including:

TfidfVectorizer: To convert text data into numerical TF-IDF features.

MultinomialNB: The core classification algorithm used for spam detection.

MinMaxScaler: To scale the engineered numerical features.

NLTK (Natural Language Toolkit): For essential text preprocessing tasks:

stopwords: To remove common words that don't add much meaning.

word_tokenize: To break text into individual words (tokens).

PorterStemmer: To reduce words to their root form.

Joblib: For efficiently saving and loading the trained machine learning models and the scaler.

Pandas: For data manipulation and analysis during model training (though not directly in the Streamlit app).

NumPy: For numerical operations, especially with feature arrays.

SciPy: Specifically scipy.sparse.hstack and scipy.sparse.csr_matrix for efficiently combining sparse TF-IDF features with dense engineered features.

🧠 How the Model Works
The spam detection model follows a typical Natural Language Processing (NLP) pipeline:

Data Collection & Preprocessing (Training Phase):

Text messages were collected and labeled as "ham" (non-spam) or "spam".

Text Cleaning: Messages were converted to lowercase, punctuation was removed, and text was tokenized.

Stop Word Removal & Stemming: Common stop words (e.g., "the", "is", "a") were removed, and words were reduced to their root forms (e.g., "running" becomes "run") using Porter Stemmer.

Feature Engineering:

TF-IDF Vectorization: The cleaned text was transformed into numerical vectors using TfidfVectorizer. This method assigns weights to words based on their frequency in a document and across the entire dataset, highlighting important terms. Your vectorizer was configured to extract 3000 features.

Additional Numerical Features: To enhance the model's ability to identify spam, four additional features were engineered from the raw text:

char_count: Total number of characters in the message.

word_count: Total number of words in the message.

punctuation_count: Total number of punctuation marks.

uppercase_word_count: Number of words written entirely in uppercase.

Feature Scaling: The engineered numerical features were scaled using MinMaxScaler to bring them into a similar range, preventing features with larger values from dominating the model.

Feature Combination:

The 3000 TF-IDF features and the 4 scaled engineered features were horizontally stacked (hstack) to form a single feature vector of 3004 dimensions for each message.

Model Training (Multinomial Naive Bayes):

A MultinomialNB classifier was trained on these combined features and the corresponding spam/ham labels. Multinomial Naive Bayes is well-suited for text classification tasks due to its effectiveness with count-based features (like TF-IDF).

The alpha parameter of the MultinomialNB model was tuned using GridSearchCV to find the optimal smoothing value for better accuracy.

Model Saving:

The fitted TfidfVectorizer, the trained MultinomialNB model, and the fitted MinMaxScaler were all saved as .pkl files (tfidf_vectorizer.pkl, spam_detector_model.pkl, scaler.pkl) using joblib. This allows the Streamlit application to load and use these components for predictions without retraining.

⚙️ Local Setup & Usage
To run this spam detection web app on your local machine:

Prerequisites
Make sure you have Python installed (Python 3.7+ is recommended).

1. Clone the Repository
First, clone this GitHub repository to your local machine:

git clone https://github.com/your-username/spam-detector-app.git
cd spam-detector-app

(Replace your-username/spam-detector-app with your actual GitHub repository path)

2. Place Model Files
Ensure the following three .pkl files are present in the root directory of your cloned repository (alongside spam_app.py):

tfidf_vectorizer.pkl

spam_detector_model.pkl

scaler.pkl

These files are crucial as they contain your trained model components.

3. Install Dependencies
Install all the required Python libraries using the requirements.txt file:

pip install -r requirements.txt

This command will install streamlit, joblib, scikit-learn, nltk, numpy, pandas, and scipy.

4. Run the Streamlit App
Once all dependencies are installed, you can launch the application:

streamlit run spam_app.py

This command will open the Streamlit application in your default web browser.

5. Use the App
In the web interface, you will see a text area.

Type or paste any message you want to classify.

Click the "Predict" button.

The app will display whether the message is "Spam" or "Not Spam".

☁️ Deployment
This application is designed for easy deployment using Streamlit Community Cloud.

Push to GitHub: Ensure all necessary files (spam_app.py, tfidf_vectorizer.pkl, spam_detector_model.pkl, scaler.pkl, requirements.txt) are pushed to a public GitHub repository.

Connect to Streamlit Community Cloud: Go to share.streamlit.io, sign in with your GitHub account, and select your repository.

Deploy: Follow the on-screen instructions to deploy your app. Streamlit will handle the environment setup and make your app accessible via a public URL.

🤝 Contributing
If you have suggestions for improvements or find any issues, feel free to open an issue or submit a pull request.

📧 Contact
For any questions or feedback, please reach out.
