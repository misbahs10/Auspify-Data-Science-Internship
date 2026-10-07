# Auspify Data Science Internship

A complete Data Science internship portfolio developed during the Auspify Technologies Data Science Internship.

This project demonstrates an end-to-end data science workflow using a Netflix titles dataset, covering data cleaning, exploratory data analysis, recommendation-system analysis, trend forecasting, machine learning classification, and an interactive business intelligence dashboard.

## Project Overview

The internship task list contains six progressively advanced data science tasks:

| Task | Project | Status |
|---|---|---|
| 1 | Data Cleaning & Preprocessing | ✅ Complete |
| 2 | Exploratory Data Analysis | ✅ Complete |
| 3 | Recommendation System Analysis | ✅ Complete |
| 4 | Trend Prediction Analysis | ✅ Complete |
| 5 | ML Classification Model | ✅ Complete |
| 6 | Business Insights Dashboard | ✅ Complete |

## Dataset

The Netflix dataset contains **8,790 titles** and 10 original columns:

- `show_id`
- `type`
- `title`
- `director`
- `country`
- `date_added`
- `release_year`
- `rating`
- `duration`
- `listed_in`

The cleaned dataset additionally contains:

- `added_year`
- `added_month`
- `duration_value`
- `duration_unit`

## Task 1 — Data Cleaning & Preprocessing

Performed:

- Dataset inspection
- Missing-value analysis
- Duplicate detection
- Standardization of inconsistent values
- Text cleaning
- Date conversion
- Date feature extraction
- Duration parsing
- Clean dataset generation

Output:

`data/netflix_cleaned.csv`

## Task 2 — Exploratory Data Analysis

Analyzed:

- Movie vs TV Show distribution
- Content categories
- Content ratings
- Country distribution
- Release-year trends
- Major dataset patterns

The analysis shows that Movies represent the majority of the catalog, while release activity changes substantially across years.

## Task 3 — Recommendation System Analysis

Built a content-based recommendation system using:

- TF-IDF vectorization
- Content feature engineering
- Cosine similarity
- Title-based recommendation function
- Recommendation evaluation

The system uses metadata such as content type, director, country, rating, and categories to identify similar titles.

Outputs:

- `outputs/task_3_recommendations.csv`
- `outputs/task_3_evaluation.csv`

## Task 4 — Trend Prediction Analysis

Analyzed annual Netflix content trends and developed a Linear Regression forecasting model.

The historical analysis identified **2018 as the peak year**, with **1,146 titles** in the dataset.

The forecasting model generated scenario-based projections for 2022–2026.

> Forecast values are model-based projections and should not be interpreted as actual future Netflix content counts.

Outputs:

- `outputs/task_4_yearly_content_trend.png`
- `outputs/task_4_historical_vs_forecast.png`
- `outputs/task_4_future_forecast.csv`

## Task 5 — Machine Learning Classification

Built classification models to predict whether a title is a **Movie** or **TV Show**.

### Models

- Logistic Regression
- Decision Tree
- Random Forest

### Features

- Release year
- Rating
- Duration value
- Duration unit
- Country

All three evaluated models achieved **100% test accuracy** on the prepared test set.

Because all three models achieved the same accuracy, Logistic Regression was selected as the preferred baseline because it provides lower model complexity.

The perfect test result should still be validated on independent external data before production deployment.

Outputs:

- `outputs/task_5_confusion_matrices.png`
- `outputs/task_5_model_accuracy_comparison.png`
- `outputs/task_5_model_evaluation.csv`
- `outputs/task_5_model_comparison.csv`

## Task 6 — Business Insights Dashboard

Task 6 integrates the previous analytical work into an interactive **Streamlit Business Intelligence Dashboard**.

### Dashboard Features

- KPI metrics
- Movie vs TV Show distribution
- Release-year trends
- Geographic analysis
- Category analysis
- Rating analysis
- Machine learning prediction
- Trend forecasting
- Data-driven business insights
- Business recommendations
- Professional business report

### Dashboard Preview

Run the application locally:

```bash
streamlit run dashboard/app.py
```

The dashboard allows users to:

1. Filter content by Movie or TV Show.
2. Explore catalog-level KPIs.
3. Analyze release-year trends.
4. Compare geographic content representation.
5. Explore category and rating distributions.
6. Enter title metadata and predict Movie vs TV Show.
7. Review model-based future trend projections.
8. Review business findings and recommendations.

## Key Business Findings

Based on the complete dataset:

- **Total titles:** 8,790
- **Movies:** 6,126
- **Movie share:** 69.69%
- **TV Shows:** 2,664
- **TV Show share:** 30.31%
- **Peak release year:** 2018
- **Peak-year titles:** 1,146
- **2021 titles:** 592

The dataset shows a decline in annual content releases after the 2018 peak.

The dashboard dynamically calculates the leading countries, categories, and ratings based on the selected filters.

## Business Recommendations

The dashboard provides several data-driven recommendations:

- Monitor the Movie and TV Show portfolio balance.
- Analyze highly represented content categories.
- Evaluate geographic markets using engagement and revenue data.
- Investigate the decline in releases after the 2018 peak.
- Use machine learning as decision support rather than a standalone production system.
- Treat forecasting results as scenarios rather than guaranteed future outcomes.
- Validate analytical models using additional real-world business data.

## Project Structure

```text
Auspify-Data-Science-Internship/
│
├── data/
│   ├── Dataset.csv
│   └── netflix_cleaned.csv
│
├── notebooks/
│   ├── Task_1_Data_Cleaning_Preprocessing.ipynb
│   ├── Task_2_Exploratory_Data_Analysis.ipynb
│   ├── Task_3_Recommendation_System_Analysis.ipynb
│   ├── Task_4_Trend_Prediction_Analysis.ipynb
│   └── Task_5_ML_Classification_Model.ipynb
│
├── outputs/
│   ├── task_3_recommendations.csv
│   ├── task_3_evaluation.csv
│   ├── task_4_yearly_content_trend.png
│   ├── task_4_historical_vs_forecast.png
│   ├── task_4_future_forecast.csv
│   ├── task_5_confusion_matrices.png
│   ├── task_5_model_accuracy_comparison.png
│   ├── task_5_model_evaluation.csv
│   └── task_5_model_comparison.csv
│
├── dashboard/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/misbahs10/Auspify-Data-Science-Internship.git
```

Move into the project:

```bash
cd Auspify-Data-Science-Internship
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

## End-to-End Workflow

```text
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Recommendation System
      ↓
Trend Forecasting
      ↓
Machine Learning Classification
      ↓
Business Intelligence Dashboard
```

## Notes

This project demonstrates an end-to-end Data Science workflow.

The forecasting and machine learning results are analytical outputs based on the available dataset. Additional external data and independent validation would be required before using these models for production business decisions.

## Author

**Misbah Sajjad**

Artificial Intelligence & Data Science

GitHub: https://github.com/misbahs10/Auspify-Data-Science-Internship.git

LinkedIn: https://www.linkedin.com/in/misbahsajjad23/