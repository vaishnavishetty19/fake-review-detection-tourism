from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# Load the SVM model and TF-IDF vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

# Create a directory to save the CSV if it doesn't exist
os.makedirs('submissions', exist_ok=True)

# Home route to render the index.html
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle review predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    reviews = data.get('reviews', [])  # Get the list of reviews

    if not reviews:
        return jsonify({'error': 'No reviews provided.'}), 400

    # Vectorize the reviews using TF-IDF
    reviews_tfidf = tfidf_vectorizer.transform(reviews)

    # Make predictions using the SVM model
    predictions = svm_model.predict(reviews_tfidf)

    # Map predictions back to labels
    label_mapping = {0: 'deceptive', 1: 'truthful'}
    predictions_labels = [label_mapping[pred] for pred in predictions]

    # Save reviews and predictions to CSV
    submissions_df = pd.DataFrame({
        'Review': reviews,
        'Prediction': predictions_labels
    })
    submissions_df.to_csv(f'submissions/reviews_predictions.csv', mode='a', header=not os.path.exists('submissions/reviews_predictions.csv'), index=False)

    # Return the predictions as JSON
    return jsonify({'predictions': predictions_labels})

if __name__ == '__main__':
    app.run(debug=True)
