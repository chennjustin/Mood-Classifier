# Instagram Comment Classification System - Text Mining Project

## Overview

This project provides a complete workflow for collecting, labeling, and classifying Instagram comments using Python. The system consists of three main components:

1. **Instagram Comment Crawler** - Automatically collects comments from specified posts
2. **Interactive Labeling Tool** - Helps categorize comments into predefined classes
3. **BERT-based Classifier** - Uses deep learning to automatically classify new comments

After finishing the model development, we can give it some simple comments and it will tell you the sentiment of the comments.

## 🏗️ Project Structure
text-mining-project/  
├── Classification/ # Classification module  
│ ├── bert.py # BERT+SVM classifier  
│ ├── chinese_stopwords.txt # Chinese stopwords  
│ ├── test_label.txt # Test set labels  
│ └── training_label.txt # Training set labels  
├── crawler/ # Crawler module  
│ ├── IG_automatic.py # Instagram crawler  
│ └── urls.txt # Target post URLs  
└── labeler/ # Labeling tool  
│ ├── label_output.txt # Labeling results  
│ └── main.py # Interactive labeling tool  

## 🛠️ Modules

## How the System Works

### 1. Data Collection
Files involved:  
- `urls.txt`: Contains Instagram post URLs (one per line) to crawl  
- `IG_automatic.py`: Main crawler script that:  
  - Reads URLs from urls.txt  
  - Logs into Instagram using provided credentials  
  - Crawls all comments from each post  
  - Saves each comment as individual .txt files (e.g., comment_1.txt, comment_2.txt)  
 
### 2. Data Labeling
Files involved:  
- `main.py`: Interactive labeling tool that:  
  - Scans the Result directory for comment_*.txt files  
  - Displays each comment for manual classification  
  - Records labels in memory  
- `label_output.txt`: Generated output file containing the mapping of:  
1 627 636 651... # Class 1 comments  
2 623 624 629... # Class 2 comments  
3 621 622 625... # Class 3 comments  

### 3. Model Training
Key files:  
- `training_label.txt`: Pre-labeled data for training (same format as label_output.txt)  
- `test_label.txt`: Held-out test set for evaluation  
- `chinese_stopwords.txt`: List of words to filter out during preprocessing  
- `bert.py`: Main classification script that:  
1. Reads comment files based on training_label.txt  
2. Preprocesses text (removes stopwords, emojis, etc.)   
3. Uses BERT to extract features   
4. Trains SVM classifier  
5. Evaluates on test set using test_label.txt  

### Instagram Comment Crawler  
- Automatic Instagram login  
- Crawl all comments from specified posts  
- Scroll to load more comments  
- Save each comment as separate txt file  

##  How to Run the System

### Step 1: Data Collection
1. Add target URLs to `crawler/urls.txt`
2. Configure credentials in `IG_automatic.py`
3. Run:
```python
python IG_automatic.py
```
### Step 2: Label Comments
1. After crawling, run:
```python
python main.py
```
2. Follow prompts to label each comment
3. Labels are saved to label_output.txt

### Step 3: Train Classifier
1. Merge new labels from label_output.txt into training_label.txt
2. Run classifier:
```python
python bert.py
```
2. Follow prompts to label each comment
3. Labels are saved to label_output.txt

### 📝 Requirements
- Python 3.8+
- transformers
- torch
- scikit-learn
- selenium
- jieba
