# 📧 Spam Detection Web App

Welcome to the **Spam Detection Web App** repository! This project features an innovative Streamlit web application designed for real-time spam detection. Utilizing a sophisticated machine learning model, the app classifies text messages (or brief emails) into two categories: "Spam" or "Not Spam" (also referred to as Ham). The user-friendly interface allows for effortless text input, providing instant predictions and enhancing the user experience.

## ✨ Features

- **Interactive Web Interface:** Built with Streamlit, the application offers a seamless experience for text input and immediate prediction display.

- **Machine Learning Powered:** The application employs a Multinomial Naive Bayes classifier, a robust algorithm known for its effectiveness in text classification.

- **Advanced Text Preprocessing:** The app incorporates TF-IDF vectorization to transform text into numerical features, enhancing the model's ability to understand and classify content accurately.

- **Enriched Feature Set:** The model's prediction accuracy is further improved by integrating engineered features, such as character count, word count, punctuation count, and uppercase word count.

- **Real-time Predictions:** Users can receive instant classification results while typing or pasting text into the application.

## 🚀 Technologies & Libraries Used

This project is powered by Python and employs a rich stack of libraries, ensuring optimal functionality. Here are the essential technologies utilized:

- **Streamlit:** The backbone of the interactive web application.

- **Scikit-learn (sklearn):** Provides core machine learning functionalities, including:
  - **TfidfVectorizer:** Converts text data into numerical TF-IDF features.
  - **MultinomialNB:** The primary classification algorithm for spam detection.
  - **MinMaxScaler:** Scales engineered numerical features for uniformity.

- **NLTK (Natural Language Toolkit):** Essential for text preprocessing tasks, including:
  - **stopwords:** Removes common words that lack significant meaning.
  - **word_tokenize:** Breaks text into individual words (tokens).
  - **PorterStemmer:** Reduces words to their root forms (e.g., "running" becomes "run").

- **Joblib:** Efficiently saves and loads trained machine learning models and the scaler.

- **Pandas:** Manages data manipulation and analysis during model training (though not directly in the Streamlit app).

- **NumPy:** Handles numerical operations, particularly with feature arrays.

- **SciPy:** Specifically uses `scipy.sparse.hstack` and `scipy.sparse.csr_matrix` to efficiently combine sparse TF-IDF features with dense engineered features.

## 🧠 How the Model Works

The spam detection model operates through a standard Natural Language Processing (NLP) pipeline:

### Data Collection & Preprocessing (Training Phase)

1. **Data Gathering:** Text messages were collected and labeled as "ham" (non-spam) or "spam".

2. **Text Cleaning:** Messages were converted to lowercase, punctuation was removed, and text was tokenized.

3. **Stop Word Removal & Stemming:** Common stop words (e.g., "the", "is", "a") were eliminated, and words were stemmed to their root forms using Porter Stemmer.

### Feature Engineering

- **TF-IDF Vectorization:** The cleaned text was transformed into numerical vectors using TfidfVectorizer, assigning weights to words based on their frequency within documents and across the dataset, thus emphasizing significant terms. The vectorizer is configured to extract 3000 features.

- **Additional Numerical Features:** To improve the model's spam identification capabilities, four extra features were generated from the raw text:
  - **char_count:** Total number of characters in the message.
  - **word_count:** Total number of words in the message.
  - **punctuation_count:** Total number of punctuation marks.
  - **uppercase_word_count:** Number of words written entirely in uppercase.

- **Feature Scaling:** The engineered numerical features were scaled using MinMaxScaler to standardize their range, ensuring that features with larger values do not overshadow others during model training.

### Feature Combination

The 3000 TF-IDF features and the 4 scaled engineered features were horizontally stacked (hstack) to create a single feature vector comprising 3004 dimensions for each message.

### Model Training (Multinomial Naive Bayes)

A MultinomialNB classifier was trained using the combined feature set and the corresponding spam/ham labels. This algorithm is particularly effective for text classification tasks that utilize count-based features like TF-IDF.

The alpha parameter of the MultinomialNB model was fine-tuned using GridSearchCV to identify the optimal smoothing value, enhancing accuracy.

### Model Saving

The fitted TfidfVectorizer, trained MultinomialNB model, and fitted MinMaxScaler were saved as .pkl files (tfidf_vectorizer.pkl, spam_detector_model.pkl, scaler.pkl) using joblib. This allows the Streamlit application to load these components efficiently for predictions without the need for retraining.

## ⚙️ Local Setup & Usage

To run this spam detection web app on your local machine, follow these steps:

### Prerequisites

Ensure that you have Python installed (Python 3.7+ is recommended).

### 1. Clone the Repository

Clone this GitHub repository to your local machine:

```bash
git clone https://github.com/your-username/spam-detector-app.git
cd spam-detector-app
```
*(Replace your-username/spam-detector-app with your actual GitHub repository path)*

### 2. Place Model Files

Ensure the following three .pkl files are present in the root directory of your cloned repository (alongside spam_app.py):

- `tfidf_vectorizer.pkl`
- `spam_detector_model.pkl`
- `scaler.pkl`

These files are critical for the app's functionality as they contain the trained model components.

### 3. Install Dependencies

Install all required Python libraries using the requirements.txt file:

```bash
pip install -r requirements.txt
```

This command will install Streamlit, joblib, scikit-learn, nltk, numpy, pandas, and scipy.

### 4. Run the Streamlit App

After installing all dependencies, launch the application:

```bash
streamlit run spam_app.py
```

This command will open the Streamlit application in your default web browser.

### 5. Use the App

In the web interface, you will see a text area where you can:

- Type or paste any message you want to classify.
- Click the "Predict" button to receive instant feedback on whether the message is "Spam" or "Not Spam".

## ☁️ Deployment

This application is designed for straightforward deployment using Streamlit Community Cloud.

### Steps to Deploy:

1. **Push to GitHub:** Ensure all necessary files (spam_app.py, tfidf_vectorizer.pkl, spam_detector_model.pkl, scaler.pkl, requirements.txt) are pushed to a public GitHub repository.

2. **Connect to Streamlit Community Cloud:** Navigate to share.streamlit.io, sign in with your GitHub account, and select your repository.

3. **Deploy:** Follow the on-screen instructions to deploy your app. Streamlit will manage the environment setup and make your app accessible via a public URL.

## 🤝 Contributing

We welcome suggestions for improvements or reports on any issues! Feel free to open an issue or submit a pull request.

## 📧 Contact

For any questions or feedback, please don’t hesitate to reach out. Your input is invaluable in enhancing this project!

Thank you for exploring the Spam Detection Web App, and we hope it serves as a useful tool in combating unwanted spam!
