Fake Review Detection System Using Machine Learning

1. Major Functionality
The system provides the following core features:

Preprocessing of Raw Text: Converts text to lowercase, tokenizes it, removes stopwords, and applies stemming/lemmatization to reduce words to their root forms.

Feature Engineering: Transforms processed text into numerical features using Count Vectorizer and TF-IDF (Term Frequency-Inverse Document Frequency). It also calculates review length.

Model Training and Evaluation: Trains and evaluates multiple classification algorithms, assessing their performance using metrics like accuracy, confusion matrix, and classification reports.

Best Model Selection and Saving: Automatically selects the best-performing model based on accuracy and saves it for future predictions.

User-Friendly Interface: Provides a simple web interface for users to input reviews and receive real-time predictions.

2. Dataset
The project utilizes a dataset sourced from Kaggle, containing product reviews with the following fields:
category: Product category (e.g., Home_and_Kitchen)
rating: Star rating (1-5)
label: Original label (CG for genuine, others for fake)
text_: The actual text of the product review

For classification, the labels were transformed:
OR (Original Review) 
CG (Computer Generated Fake Review) 

Preprocessing steps were applied to clean the text data, and any empty values were removed to ensure model quality.

3. Model Comparison & Results
Six different machine learning models were trained and rigorously compared for their effectiveness in fake review classification. The models were:

•Multinomial Naive Bayes (NB)
•Random Forest (RF)
•Decision Tree (DT)
•K-Nearest Neighbors (KNN)
•Support Vector Machine (SVM)
•Logistic Regression (LR)

Conclusion: The Support Vector Machine (SVM) model demonstrated the highest accuracy at 87.80%, making it the best-performing model for this classification task.

4. Tech Stack
Frontend: Streamlit
Backend: Python
Machine Learning Libraries: Scikit-learn, NLTK
Data Manipulation: Pandas, NumPy
Visualization: Matplotlib

5. User Interface
The system features a simple and intuitive web-based user interface built with Streamlit, allowing users to easily input a product review and receive an immediate prediction (Genuine Review or Fake Review).

6. Installation & Setup
To set up and run this project locally, follow these steps:

Clone the repository:

git clone <https://github.com/maryamzia4/fake-review-detection>
cd fake-review-detector

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install the required libraries:

pip install pandas numpy nltk scikit-learn matplotlib streamlit

(You might need to download NLTK data: python -m nltk.downloader all or specific modules like punkt, stopwords, wordnet.)

Dataset: Ensure your dataset (e.g., fake_reviews_dataset.csv) is placed in the appropriate directory as expected by the project's code. You can download it from the Kaggle Dataset Source.

7. Usage
To run the Streamlit web application:

Navigate to the project's root directory in your terminal.

Run the Streamlit application:

streamlit run app.py

The application will open in your default web browser (usually at http://localhost:8501).


Dataset Source:
[Kaggle Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)

8. Authors
Maryam Zia 
Humma Laila 

9. License
This project is open-source and available under the MIT License.