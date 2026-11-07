# Credit Card Customer Segmentation

An end-to-end customer segmentation project using **K-Means
clustering**, Python, scikit-learn, and Streamlit. The project analyses
credit-card customer behaviour and groups customers with similar
spending and payment patterns. An interactive dashboard supports both
prediction with a previously trained model and fresh clustering on a
newly uploaded compatible dataset.

## Project Objectives

-   Explore and understand credit-card customer behaviour.
-   Clean, transform, and scale customer-level features for clustering.
-   Identify customer segments using K-Means.
-   Evaluate candidate cluster counts using the Elbow Method and
    Silhouette Score.
-   Profile clusters to understand differences in spending and payment
    behaviour.
-   Provide an interactive dashboard for existing-model predictions and
    new-dataset clustering.

## Dataset

The project uses the **Credit Card Dataset for Clustering** from Kaggle:

-   Source: [Credit Card Dataset for Clustering --
    Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
-   Original size: **8,950 customer records and 18 columns**.
-   `CUST_ID` is retained for identification and reporting, but is not
    used as a clustering feature.
-   The clustering pipeline uses **16 customer-behaviour features**.
    `PURCHASES` is excluded from the selected feature set; the remaining
    selected features are listed below.

### Features used for clustering

1.  `BALANCE`
2.  `BALANCE_FREQUENCY`
3.  `ONEOFF_PURCHASES`
4.  `INSTALLMENTS_PURCHASES`
5.  `CASH_ADVANCE`
6.  `PURCHASES_FREQUENCY`
7.  `ONEOFF_PURCHASES_FREQUENCY`
8.  `PURCHASES_INSTALLMENTS_FREQUENCY`
9.  `CASH_ADVANCE_FREQUENCY`
10. `CASH_ADVANCE_TRX`
11. `PURCHASES_TRX`
12. `CREDIT_LIMIT`
13. `PAYMENTS`
14. `MINIMUM_PAYMENTS`
15. `PRC_FULL_PAYMENT`
16. `TENURE`

For Method 1 and Method 2, an uploaded dataset must contain these same
16 features with compatible meanings and units. `CUST_ID` is optional
and is not used for clustering.

## Project Workflow

1.  **Exploratory Data Analysis:** inspect the data, missing values,
    distributions, skewness, correlations, and potential outliers.
2.  **Preprocessing:** select the clustering features, impute missing
    values, apply `log1p` to selected skewed monetary/transaction
    features, and standardise the features.
3.  **Cluster evaluation:** compare candidate values of K using inertia
    (Elbow Method) and Silhouette Score.
4.  **Model training:** train a K-Means model and assign a cluster to
    each customer.
5.  **Cluster profiling:** compare customer-behaviour feature profiles
    across the resulting clusters.
6.  **Dashboard:** explore existing assignments, predict with the saved
    model, or run a fresh clustering workflow on a compatible uploaded
    dataset.

## Methodology

### Data preprocessing

The original preprocessing pipeline includes:

-   Median imputation for missing numerical values.
-   `log1p` transformation for selected non-negative, skewed monetary
    and transaction-count features.
-   `StandardScaler` to standardise the feature values.
-   Exclusion of `CUST_ID` from model training.

The fitted preprocessing artifacts are saved so that Method 1 can apply
the same transformations to uploaded data before prediction.

### K-Means clustering

The original model was trained with:

-   Number of clusters: **5**
-   Initialisation: `k-means++`
-   `n_init`: `10`
-   Maximum iterations: `300`
-   `random_state`: `42`

Candidate cluster counts from K=2 to K=10 were evaluated. The final
original model uses K=5, selected as a practical balance between the
Elbow curve, sampled Silhouette Score, cluster distribution, and
usefulness of behavioural profiles.

### Original model evaluation

  Metric                                                 Result
  ------------------------ ------------------------------------
  Customers                                               8,950
  Clustering features                                        16
  Selected clusters                                           5
  Inertia                                           75,294.2586
  Silhouette Score                                       0.2242
  Silhouette calculation     Sample of 2,000, `random_state=42`
  K-Means iterations                                         56

The Silhouette Score is a sample-based estimate, not a full-dataset
score. A relatively modest score indicates that customer behaviours
overlap; cluster profiles should therefore be interpreted carefully.
K-Means cluster IDs are identifiers, not pre-existing customer
categories.

## Streamlit Dashboard

Run the application to access two workflows.

### Method 1 -- Existing Model Prediction

-   Upload a compatible customer CSV.
-   Validate the required feature columns and input values.
-   Apply the saved imputer, transformations, and scaler.
-   Use the saved K-Means model to assign customers to the original five
    clusters.
-   View the prediction summary, cluster distribution, and
    customer-level assignments.
-   Download the prediction results as a CSV.

When the uploaded file is recognised as the original dataset with
matching customer IDs and row order, the dashboard can compare the
predicted assignments with the stored original assignments. A 100% match
was observed during the original-dataset validation.

### Method 2 -- New Dataset Clustering

-   Upload a compatible customer CSV containing the same 16 features.
-   Fit a new median imputer and scaler on the uploaded dataset.
-   Apply the selected `log1p` transformations and standardisation.
-   Evaluate candidate K values using inertia and Silhouette Score.
-   Select a K value and train a fresh K-Means model.
-   View cluster distribution, median behavioural profiles, and customer
    assignments.
-   Download the new clustering results as a CSV.

Method 2 discovers clusters within the uploaded dataset. Its cluster
labels are specific to that newly trained model and do not necessarily
correspond to the labels from Method 1. The K evaluation is decision
support; the highest Silhouette Score does not automatically guarantee
the most useful business segmentation.

## Example Business Interpretation

Depending on the observed cluster profiles, segments may show patterns
such as:

-   Customers who use cash advances more frequently.
-   Customers who make regular instalment purchases.
-   Customers with higher one-off purchase activity.
-   Customers with relatively low card activity.
-   Customers with a mixture of purchase and cash-advance behaviour.

These are behavioural descriptions, not credit-risk ratings. The model
does **not** predict loan default, fraud, future spending, or credit
scores.

## Repository Structure

``` text
Credit-Card-Customer-Segmentation/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── CC GENERAL.csv
│   └── processed/
│       ├── credit_card_customer_features_scaled.csv
│       ├── customer_segments.csv
│       └── method1_test_sample.csv
│
├── models/
│   ├── final_customer_segmentation_bundle.joblib
│   └── preprocessing_artifacts.joblib
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_kmeans_clustering.ipynb
│
├── outputs/
│   ├── plots/
│   │   ├── kmeans_cluster_evaluation.png
│   │   └── pca_customer_segmentation.png
│   └── reports/
│       └── customer_cluster_profiles.xlsx
│
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

The exact files under `src/` and any additional output files may vary as
the project evolves.

## Installation and Usage

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Credit-Card-Customer-Segmentation
```

### 2. Create and activate a virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Obtain the dataset and model artifacts

Download the dataset from the Kaggle source linked above and place it
at:

``` text
data/raw/CC GENERAL.csv
```

The dashboard also expects the trained model and preprocessing artifacts
in the `models/` directory. If these files are not included in the
repository, obtain or generate them before launching the dashboard.

### 5. Run the Streamlit application

From the project root:

``` bash
streamlit run app/streamlit_app.py
```

Open the local URL shown in the terminal, usually
`http://localhost:8501`.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   Joblib
-   Jupyter Notebook
-   Streamlit

## Limitations

-   K-Means requires a chosen number of clusters; evaluation metrics
    support but do not prove the choice.
-   Customer segments can overlap, as reflected by the Silhouette Score.
-   Method 1 requires compatible feature definitions and units because
    it uses the original fitted preprocessing pipeline and model.
-   Method 2 trains new preprocessing objects and a new model on the
    uploaded data; its cluster IDs are not directly comparable with
    Method 1.
-   Cluster interpretation is descriptive and does not establish
    causation or financial risk.
-   The dashboard's successful test on a sample drawn from the original
    dataset verifies the workflow, but is not an independent external
    validation.

## Author

**K A Kaushik**

## Acknowledgements

-   Dataset: [Arjun Bhasin -- Credit Card Dataset for
    Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
-   Libraries and tools: Python, Pandas, NumPy, Scikit-learn,
    Matplotlib, Seaborn, Jupyter, and Streamlit.
