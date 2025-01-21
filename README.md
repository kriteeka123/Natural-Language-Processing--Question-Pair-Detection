Overview 
This project focuses on detecting duplicate or semantically similar questions using Natural Language Processing (NLP) and Deep Learning. The goal is to classify whether two given questions have the same intent or meaning. This is particularly useful for platforms like Quora, Stack Overflow, and search engines to improve user experience by avoiding redundant content.  

Objective 
- Build a model to identify duplicate questions based on their textual similarity.  
- Utilize word embeddings and deep learning models for classification.  
- Improve accuracy using advanced NLP techniques like LSTMs, GRUs, or Transformers.  

Dataset  
- Quora Question Pairs Dataset (or any other labeled dataset with duplicate questions).  
- Contains pairs of questions with labels (1: Duplicate, 0: Not Duplicate).  

Approach  
1. Data Preprocessing
   - Tokenization, Lowercasing, Stopword Removal, Lemmatization  
   - Word embeddings (Word2Vec, GloVe, BERT)  

2. Feature Engineering  
   - Cosine Similarity, TF-IDF, Jaccard Similarity  
   - Sentence Embeddings  

3. Modeling 
   - Traditional ML models (Logistic Regression, SVM, Random Forest)  
   - Deep Learning models (LSTMs, BiLSTMs, GRUs, Transformers)  
   - Fine-tuning BERT for sentence pair classification  

4. Evaluation Metrics  
   - Accuracy, Precision, Recall, F1-score  
   - AUC-ROC Curve  

Technologies Used  
- Programming Language: Python  
- Libraries: TensorFlow / PyTorch, Scikit-learn, NLTK, Spacy, Hugging Face Transformers  
- Deep Learning: LSTMs, GRUs, BERT-based models  

Installation & Setup  
1. Clone the repository:  
   git clone https://github.com/your-username/question-pair-detection.git
   cd question-pair-detection
  
2. Install dependencies:  
   pip install -r requirements.txt

3. Run the training script:  
   python train.py

4. Evaluate the model:  
   python evaluate.py


Results & Insights 
- Achieved highest accuracy using XGBClassifier.   
- Challenges: Handling paraphrasing, complex sentence structures, and ambiguous meanings.  

Future Enhancements  
- Experiment with larger transformer models (RoBERTa, T5, GPT-based architectures).  
- Implement semi-supervised learning for better generalization.  
- Deploy the model using Flask / FastAPI for real-world applications.  

