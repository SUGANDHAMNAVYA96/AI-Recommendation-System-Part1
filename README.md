# AI-Recommendation-System-Part1
## Collaborative Filtering using MovieLens 100k

## Overview
This project implements a collaborative filtering recommendation system using the MovieLens 100k dataset. The goal is to explore user–item interactions, preprocess rating data, and build a matrix factorization model to predict user preferences and generate personalized movie recommendations.

The workflow follows a standard machine learning pipeline:
data exploration → preprocessing → baseline modeling → hyperparameter tuning → evaluation → Top-N recommendations.

---

## Dataset
**MovieLens 100k**
- 100,000 ratings
- 943 users
- 1,682 movies
- Rating scale: 1–5

The dataset is converted into a pandas DataFrame and prepared for collaborative filtering.

---

## Methods

### Data Preprocessing
- Remove duplicates and missing values
- Verify correct data types
- Drop unused timestamp column
- Filter extremely inactive users/items to reduce sparsity
- Analyze rating distributions and interaction patterns

### Modeling
- Train/test split (75% / 25%)
- Collaborative filtering using Singular Value Decomposition (SVD)
- Hyperparameter tuning with GridSearchCV
- Evaluation using Root Mean Squared Error (RMSE)
- Generate Top-N personalized recommendations

### Libraries
- pandas
- numpy
- matplotlib / seaborn
- scikit-learn
- surprise (recommender systems)

---

## Results
| Model | RMSE |
|-------|--------|
| Baseline SVD | 0.948 |
| After filtering sparse users/items | 0.936 |
| Tuned SVD (GridSearch) | **0.918** |

Performance improved through preprocessing and hyperparameter optimization.

---

## Installation
Install dependencies:

```bash
pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn
```

---

## How to Run
Open the notebook and run all cells:

```
GroupProject_Part1_RecommendationSystem.ipynb
```

This will reproduce:
- preprocessing
- model training
- evaluation
- Top-N recommendations

---

## Outputs
- Rating distribution visualizations
- Sparsity analysis
- RMSE performance metrics
- Top-N movie recommendations per user

---

## Authors
Marisa Tania  
Navya Sugandham
