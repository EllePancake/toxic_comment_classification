### Toxic Comment Classification Project

#### **Project Overview:**
This project focuses on building a **multi-label classifier** to automatically identify and flag toxic comments. The dataset used comes from the [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). The objective is to classify user comments into one or more of six possible categories: toxic, severe toxic, obscene, threat, insult, and identity hate.

The project is designed to simulate the work of content moderators who need to identify and flag inappropriate content on online platforms. Manually monitoring content is expensive and impractical, so the model aims to automate this task by classifying toxic comments, helping platforms take appropriate actions.

#### **Notebooks:**
1. [**Exploratory Data Analysis (EDA)**](https://github.com/EllePancake/toxic_comment_classification/blob/main/toxic%20comment%20project%20-%20EDA.ipynb)
2. [**Model One**](https://github.com/EllePancake/toxic_comment_classification/blob/main/toxic%20comment%20project%20-%20model%20one.ipynb)
3. [**Models Two and Three**](https://github.com/EllePancake/toxic_comment_classification/blob/main/toxic%20comment%20project%20-%20models%20two%20and%20three.ipynb)
4. [**Model Four**](https://github.com/EllePancake/toxic_comment_classification/blob/main/toxic%20comment%20project%20-%20model%20four.ipynb)

---

#### **Project Steps:**

1. **Exploratory Data Analysis (EDA)**:
   - **Objective**: Understand the data structure, text characteristics, and class distribution.
   - **Tasks**:
     - Analyze the text data and investigate the distribution of labels.
     - Visualize class imbalances.
   - **Outcome**: Insights about the dataset, guiding the preprocessing and model-building steps.

2. **Data Preprocessing**:
   - **One-Hot Encoding**: Convert labels into one-hot encoding for multi-label classification.
   - **Text Tokenization**: Tokenize the text data using the HuggingFace `DistilBERTTokenizer`.
   - **Train/Validation/Test Split**: The dataset was split into training, validation, and test sets for model evaluation.

3. **Modeling**:
   - **DistilBERT for Sequence Classification**: HuggingFace's `DistilBERT` is used as the base transformer model for text classification.
   - **Sigmoid Activation Layer**: Applied to the output to predict independent probabilities for each class.
   - **Loss Function**: Binary Cross-Entropy Loss is used to optimize the model for multi-label classification.

   **Model Versions**:
   - **Model One**: Initial model to benchmark performance.
   - **Model Two**: Introduced class weights to handle class imbalance.
   - **Model Three**: Implemented focal loss to focus on hard-to-classify examples and mitigate the effects of class imbalance.
   - **Model Four**: Used the same architecture as Model Three but focused on tuning the learning rate to further optimize performance.

4. **Learning Rate Testing**:
   - **Goal**: Test various learning rates across a small number of epochs to identify the most effective learning rate for the model.
   - **Outcome**: After testing learning rates ranging from `1e-6` to `5e-3`, it was found that **Model 3** with a learning rate of **5e-5** performed the best, achieving the lowest average loss across all epochs.

5. **Evaluation and Testing**:
   - **Thresholding**: Predictions were thresholded at 0.5 to convert probabilities into binary classifications.
   - **Performance Metrics**: Precision, Recall, F1-Score, and ROC-AUC were used to assess model performance on both validation and test datasets.
   - **Model 3 Results**: After testing the final model with the learning rate of **5e-5**, it showed significantly better results than the other learning rates tested, so further evaluation of other learning rates was not pursued.

---

#### **Key Results and Insights:**
- **Model Performance**:
  - **Best Model**: Model 3, using a learning rate of `5e-5` with focal loss, demonstrated the best overall performance in terms of average loss reduction and classification metrics.
  - **Handling Class Imbalance**: Focal loss significantly improved the model's ability to classify underrepresented classes, especially `severe toxic` and `identity hate`.
    
- **Learning Rate Evaluation**:
  - Multiple learning rates were tested, but the model with a learning rate of `5e-5` clearly outperformed all others, leading to the conclusion that this rate is optimal for this task.
