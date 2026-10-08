# Campus Placement Analytics & Predictor

End-to-end data analysis and machine learning project on 10,000 student placement records: clean the data, find what drives placements, predict outcomes, and explore it in an interactive app.

## Live demos
- **Analytics dashboard:** https://claude.ai/artifact/QTqfJzxoYvACtD7iBMH1bi
- **Placement Predictor (what-if tool):** https://claude.ai/artifact/Dra17iaiznE2mcnWTMLhCZ

**Start here:** open `Placement_Analysis_and_Prediction.ipynb` for the complete walkthrough with charts.

## Key findings
- **Branch drives salary:** CSE averages 12.55 LPA vs Civil 5.73 LPA (about 2.2x).
- **Internships are the biggest lever:** 49% placed with 0 internships vs 90% with 2.
- **CGPA matters:** placement chance rises from 41% (CGPA below 6) to 93% (above 9).
- **Two firms dominate:** Accenture and TCS give about 59% of all placements.
- **Model insight:** technical score (26%) and internships (23%) are the top predictors. Branch adds only 1-2% once these are known.

## Model results (20% test set)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 84.1% | 87.7% | 90.3% | 89.0% | 0.899 |
| Random Forest | 84.0% | 85.1% | 93.9% | 89.3% | 0.898 |

5-fold cross-validation accuracy is about 85%.

## Salary prediction (placed students only)

| Model | MAE (LPA) | R2 |
|---|---|---|
| Linear Regression | 3.90 | 0.487 |
| Gradient Boosting | 2.50 | 0.743 |
| **Random Forest** | **2.14** | **0.798** |

Baseline (always predict the median, 8.31 LPA) has MAE 5.18 LPA. CGPA (41%) and technical score (31%) drive salary; internships matter more for getting placed than for how much you earn. Top companies pay about 25 LPA, which tree models capture but linear regression cannot.

## Workflow
1. **Load and inspect:** one CSV, 10,000 rows x 13 columns, no merge needed.
2. **Data quality audit:** no duplicates; `company` is empty only for unplaced students; salary 0 was a placeholder.
3. **Cleaning:** standardised branch names, set unplaced salary to missing, added `salary_lpa`.
4. **Analysis:** average package by branch, top recruiters, placement rate by branch, plus CGPA, internship and skill effects.
5. **Placement model:** Logistic Regression and Random Forest, with feature importance.
6. **Salary model:** Linear Regression, Random Forest and Gradient Boosting, evaluated with MAE and R2.
7. **App:** in-browser predictor with per-factor contributions and "what-if" suggestions.

## Project structure
```
placement_analysis.py        # cleaning + analysis + dashboard chart
placement_prediction.py      # placement classification + feature importance
salary_prediction.py         # salary regression models
Placement_Analysis_and_Prediction.ipynb  # full story in one notebook
student_placement_clean.csv  # cleaned dataset
placement_dashboard.png      # static dashboard
feature_importance.png       # placement drivers
salary_model.png             # salary model results
```

## Tech stack
Python, Pandas, NumPy, Seaborn, Matplotlib, scikit-learn, Chart.js, HTML/CSS/JavaScript

## Run it
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
python placement_analysis.py
python placement_prediction.py
python salary_prediction.py
```

## Limitations
The dataset is a synthetic Kaggle dataset ("Student Placement Details"), so patterns may differ from real colleges. It has no date column, so year-over-year trends were replaced by branch-wise placement rate.

## Author
Aryan Verma, B.Tech CSE (AI & ML)

- GitHub: https://github.com/av7170393-ctrl
- LinkedIn: https://www.linkedin.com/in/aryan-verma-5360a2415

