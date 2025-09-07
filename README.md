🏨 Fake Review Detection in Tourism

This project detects fake vs genuine tourism reviews using Natural Language Processing (NLP) and Machine Learning.
By analyzing review text, we extract linguistic patterns and train models (SVM, Logistic Regression, Naive Bayes) to classify whether a review is Truthful or Deceptive.
A simple Flask web app is included to let users paste a review and instantly check its authenticity.

✨ Features

Text Preprocessing → cleaning, tokenization, stopword removal, stemming.

Feature Engineering → TF-IDF vectorization of up to 5k terms.

Machine Learning Models → SVM (best performer), Logistic Regression, Multinomial Naive Bayes.

Class Imbalance Handling → SMOTE oversampling for fair training.

Deployment → Flask app for interactive review prediction.

📂 Project Structure
fake-review-detection-tourism/
├─ app/                 # Flask app
├─ src/                 # Training, inference, preprocessing code
├─ data/                # Raw & processed data (not committed)
├─ models/              # Saved models (joblib)
├─ notebooks/           # Jupyter experiments (optional)
├─ tests/               # Smoke tests
├─ requirements.txt
├─ README.md
├─ LICENSE
└─ .gitignore

🚀 Quickstart

Clone repo

git clone https://github.com/<your-username>/fake-review-detection-tourism.git
cd fake-review-detection-tourism


Create venv (optional)

python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Add dataset
Download the Ott Deceptive Opinion Spam Corpus v1.4 and place it like:

data/raw/op_spam_v1.4/positive_polarity/deceptive_from_MTurk/...
data/raw/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/...


Train models

python -m src.train


Run Flask app

python app/app.py


Visit → http://127.0.0.1:5000

🧪 Models & Results
Model	Accuracy
SVM (linear)	~90%
Logistic Regression	~86%
Naive Bayes	~82%

Best model: SVM

Artifacts saved under /models:

svm_model.pkl

tfidf_vectorizer.pkl

🧰 Tech Stack

Python 3.11

Pandas, NumPy, Scikit-learn

NLTK, Imbalanced-Learn

Flask (for app deployment)
