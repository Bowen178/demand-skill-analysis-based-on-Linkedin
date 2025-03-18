## **Demand-Skill Analysis Based on LinkedIn Job Postings**
This project aims to analyze job postings from LinkedIn, identify in-demand skills, and predict salary ranges using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. The demo application allows users to enter a job description and receive predicted **minimum and maximum salary estimates** based on a trained **LightGBM model** using **TF-IDF** vectorization.

---

### ** Features**
- **Job Description Analysis**: Extracts meaningful insights from job descriptions.
- **Salary Prediction**: Predicts **minimum and maximum salary** using machine learning.
- **Machine Learning Models**: Compares TF-IDF, Word2Vec, and GloVe embeddings combined with different models (e.g., Logistic Regression, Random Forest, LightGBM).
- **Web Demo**: Flask-based app for salary prediction based on job descriptions.



---

## ** How to Run the Demo**
### ** Setup the Environment**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/demand-skill-analysis-based-on-LinkedIn.git
   cd demand-skill-analysis-based-on-LinkedIn
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # On Mac/Linux
   env\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### ** Run the Flask App**
Once the dependencies are installed, run the Flask app:
```bash
python app.py
```
If running locally, the app will be available at:
```
http://127.0.0.1:5000/
```

---

### ** How to Use the Web App**
1. **Enter Job Description**: Type in or paste the job description in the input box.
2. **Click Predict**: The app will process the input and display the **predicted max and min salary** based on the trained **LightGBM model**.
3. **View Results**: The predicted salary range will be displayed on the webpage.

---

## **Model Training**
The following steps were used to train the salary prediction model:

1. **Data Preprocessing**
   - Tokenization, stopword removal, lemmatization.
   - Standardizing salary data (converting hourly and yearly salaries to monthly salaries).

2. **Text Vectorization**
   - Compared **TF-IDF**, **Word2Vec**, and **GloVe** embeddings.
   - TF-IDF performed best and was used in the final model.

3. **Model Selection**
   - Tried **Logistic Regression**, **Random Forest**, **LightGBM**, and **XGBoost**.
   - **LightGBM + TF-IDF** had the **highest AUC & F1-score**.
   - Trained **separate models** for **max salary** and **min salary** predictions.

---

## **Future Improvements**
- Add more **deep learning models** such as **BERT embeddings** for better predictions.
- Allow users to **upload job postings** in bulk and analyze trends.
- Improve UI with interactive **visualizations** of salary distributions.

---

## **Dependencies**
Ensure you have the following installed:
```
Flask
pandas
numpy
scikit-learn
lightgbm
gensim
matplotlib
seaborn
nltk
```
You can install them using:
```bash
pip install -r requirements.txt
```

---

## ** Acknowledgment**
- **Dataset**: [LinkedIn Job Postings](https://www.kaggle.com/) 📊
- **NLP Techniques**: TF-IDF, Word2Vec, GloVe
- **ML Models**: LightGBM, Logistic Regression, Random Forest

---
This README provides a clear guide on how to **set up, run, and understand** the demo app. Let me know if you need any modifications! 
