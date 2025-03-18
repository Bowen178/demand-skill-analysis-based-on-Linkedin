import re
import pickle
from flask import Flask, request, jsonify, render_template
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


with open("lightgbm_model_max.pkl", "rb") as model_max_file:
    lgbm_model_max = pickle.load(model_max_file)

with open("lightgbm_model_min.pkl", "rb") as model_min_file:
    lgbm_model_min = pickle.load(model_min_file)

with open("tfidf_vectorizer.pkl", "rb") as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

app = Flask(__name__)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'<[^>]+>', ' ', text) 
    text = re.sub(r'[^a-z\s]', ' ', text) 
    tokens = text.split()  
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]  
    return " ".join(tokens)  

@app.route('/')
def home():
    return render_template("index.html") 


@app.route('/predict', methods=['POST'])
def predict():
    try:
        job_desc = request.form.get("job_description")

        clean_job_desc = preprocess_text(job_desc)

        job_desc_tfidf = tfidf_vectorizer.transform([clean_job_desc])

        pred_max_salary = lgbm_model_max.predict(job_desc_tfidf)[0]  
        pred_min_salary = lgbm_model_min.predict(job_desc_tfidf)[0]  

        return jsonify({
            "max_salary": round(pred_max_salary, 2),
            "min_salary": round(pred_min_salary, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

