import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
import joblib
import warnings
from flask import Flask, request, render_template_string

# Suppress sklearn UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Load dataset
try:
    df = pd.read_csv("train.csv")
except Exception as e:
    print("Error loading CSV:", e)
    exit()

# Train model
try:
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    X_test_tfidf = vectorizer.transform(X_test)
    print("Classification report:\n", classification_report(y_test, model.predict(X_test_tfidf)))
    print("Confusion matrix:\n", confusion_matrix(y_test, model.predict(X_test_tfidf)))

    # Save model and vectorizer
    joblib.dump((vectorizer, model), "models/tfidf_logreg.joblib")
    print("Model saved successfully!")

except Exception as e:
    print("Error during training:", e)
    exit()

# Flask app
app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <title>Fake News Detector</title>
  </head>
  <body>
    <h2>Fake News Detector</h2>
    <form method="POST">
      <textarea name="news_text" rows="5" cols="60" placeholder="Paste news here..."></textarea><br><br>
      <input type="submit" value="Predict">
    </form>
    {% if prediction %}
    <h3>Prediction: {{ prediction }}</h3>
    <p>Confidence: {{ confidence }}%</p>
    {% endif %}
  </body>
</html>
"""

# Load trained model
try:
    vectorizer, model = joblib.load("models/tfidf_logreg.joblib")
except:
    print("Cannot load model: Model not found. Please run the script to train first.")
    exit()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    if request.method == "POST":
        news = request.form["news_text"]
        news_tfidf = vectorizer.transform([news])
        pred = model.predict(news_tfidf)[0]
        pred_prob = model.predict_proba(news_tfidf).max() * 100
        prediction = pred
        confidence = round(pred_prob, 2)
    return render_template_string(HTML_TEMPLATE, prediction=prediction, confidence=confidence)

# at the end of main.py (replace any existing app.run line)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))   # Render provides PORT via env
    # Turn off debug for production
    app.run(host="0.0.0.0", port=port, debug=False)


