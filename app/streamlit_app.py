import streamlit as st
import pandas as pd
import joblib
import numpy as np
import hashlib
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "models" /
    "final_customer_segmentation_bundle.joblib"
)

SEGMENT_PATH = (
    PROJECT_ROOT / "data" / "processed" /
    "customer_segments.csv"
)



# --------------------------------------------------
# HELPER: EVALUATE K-MEANS CLUSTER COUNTS FOR METHOD 2
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def evaluate_new_dataset_clusters(scaled_features, max_k):
    """Evaluate candidate K values on a newly uploaded dataset."""
    evaluation_rows = []

    for k in range(2, max_k + 1):
        candidate_model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42
        )

        labels = candidate_model.fit_predict(scaled_features)
        number_of_labels = len(np.unique(labels))

        # Silhouette is defined only when there are at least two
        # clusters and fewer clusters than observations.
        if 2 <= number_of_labels < len(scaled_features):
            sample_size = min(2000, len(scaled_features))
            try:
                score = silhouette_score(
                    scaled_features,
                    labels,
                    sample_size=sample_size,
                    random_state=42
                )
            except ValueError:
                # A sampled subset can occasionally omit a cluster.
                score = np.nan
        else:
            score = np.nan

        evaluation_rows.append({
            "K": k,
            "Inertia": float(candidate_model.inertia_),
            "Silhouette Score": float(score) if not np.isnan(score) else np.nan
        })

    return pd.DataFrame(evaluation_rows)


# --------------------------------------------------
# 3. LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_customer_data(path):
    return pd.read_csv(path)


# Check file availability
if not MODEL_PATH.exists():
    st.error("Final K-Means model not found.")
    st.stop()

if not SEGMENT_PATH.exists():
    st.error("Customer segmentation dataset not found.")
    st.stop()

model_bundle = load_model(str(MODEL_PATH))
customer_data = load_customer_data(str(SEGMENT_PATH))

# --------------------------------------------------
# 4. DASHBOARD HEADER
# --------------------------------------------------

st.title("Credit Card Customer Segmentation")
st.write(
    "An interactive dashboard for analysing customer "
    "spending behaviour using K-Means clustering."
)

st.divider()

# --------------------------------------------------
# 5. KEY PERFORMANCE INDICATORS
# --------------------------------------------------

total_customers = len(customer_data)
total_clusters = model_bundle["n_clusters"]
total_features = len(model_bundle["selected_features"])

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Customer Segments", total_clusters)
col3.metric("Clustering Features", total_features)

st.divider()

# --------------------------------------------------
# 6. CUSTOMER CLUSTER DISTRIBUTION
# --------------------------------------------------

st.subheader("Customer Segment Distribution")

cluster_distribution = (
    customer_data["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .to_frame("Customer Count")
)

cluster_distribution["Percentage (%)"] = (
    cluster_distribution["Customer Count"]
    / total_customers * 100
).round(2)

left, right = st.columns([1.5, 1])

with left:
    st.bar_chart(cluster_distribution["Customer Count"])

with right:
    st.dataframe(
        cluster_distribution,
        width="stretch"
    )

# --------------------------------------------------
# 7. CUSTOMER SEGMENT EXPLORER
# --------------------------------------------------

st.divider()

st.subheader("Customer Segment Explorer")

selected_cluster = st.selectbox(
    "Select a customer segment",
    sorted(customer_data["Cluster"].unique())
)

filtered_customers = customer_data[
    customer_data["Cluster"] == selected_cluster
]

st.write(
    f"Customers belonging to Cluster {selected_cluster}: "
    f"**{len(filtered_customers):,}**"
)

st.dataframe(
    filtered_customers,
    width="stretch",
    hide_index=True
)

# --------------------------------------------------
# 8. EXISTING MODEL - NEW CUSTOMER SEGMENTATION
# --------------------------------------------------

st.divider()

st.header("Predict Customer Segments")
st.write(
    "Upload a compatible credit card customer dataset "
    "to assign customers to the five existing segments."
)

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"],
    key="existing_model_upload"
)

if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Dataset Preview")
        st.dataframe(uploaded_df.head(), width="stretch")

        # Retrieve saved preprocessing information
        preprocessing = model_bundle["preprocessing_artifacts"]
        required_features = model_bundle["selected_features"]
        log_features = preprocessing["log_features"]

        # Check required columns
        missing_features = [
            col for col in required_features
            if col not in uploaded_df.columns
        ]

        if missing_features:

            st.error(
                "Required features are missing: "
                + ", ".join(missing_features)
            )

        elif uploaded_df.empty:

            st.error("The uploaded dataset contains no customer records.")

        else:

            # Select features in the exact training order
            input_features = uploaded_df[required_features].copy()

            # Convert values to numeric
            input_features = input_features.apply(
                pd.to_numeric,
                errors="coerce"
            )

            # Replace infinite values with missing values
            input_features = input_features.replace(
                [np.inf, -np.inf],
                np.nan
            )

            # Check for completely empty columns
            empty_columns = input_features.columns[
                input_features.isna().all()
            ].tolist()

            if empty_columns:

                st.error(
                    "These required columns contain no usable values: "
                    + ", ".join(empty_columns)
                )

            else:

                # Validate non-negative features before log transformation
                negative_features = [
                    col for col in log_features
                    if (input_features[col].dropna() < 0).any()
                ]

                if negative_features:

                    st.error(
                        "Negative values found in features requiring "
                        "log transformation: "
                        + ", ".join(negative_features)
                    )

                else:

                    # Step 1: Apply saved median imputer
                    imputed_array = preprocessing["imputer"].transform(
                        input_features
                    )

                    imputed_df = pd.DataFrame(
                        imputed_array,
                        columns=required_features,
                        index=input_features.index
                    )

                    # Step 2: Apply saved log transformation
                    imputed_df[log_features] = np.log1p(
                        imputed_df[log_features]
                    )

                    # Step 3: Apply saved standard scaler
                    scaled_array = preprocessing["scaler"].transform(
                        imputed_df[required_features]
                    )

                    # Step 4: Predict customer clusters
                    scaled_df = pd.DataFrame(
                        scaled_array,
                        columns=required_features,
                        index=input_features.index
                    )

                    predictions = model_bundle["kmeans_model"].predict(
                        scaled_df
                    )

                    # --------------------------------------------------
                    # PREDICTION VALIDATION
                    # --------------------------------------------------

                    print("\n" + "=" * 45)
                    print("EXISTING MODEL PREDICTION")
                    print("=" * 45)

                    print(f"Customers processed: {len(predictions):,}")
                    print(f"Clusters identified: {len(np.unique(predictions))}")

                    # Check whether this is the original dataset
                    # with identical customer order and feature values

                    is_original_dataset = False

                    if (
                        "CUST_ID" in uploaded_df.columns
                        and "CUST_ID" in customer_data.columns
                        and len(uploaded_df) == len(customer_data)
                    ):

                        # Check customer IDs and their order
                        same_customer_ids = (
                            uploaded_df["CUST_ID"]
                            .reset_index(drop=True)
                            .equals(
                                customer_data["CUST_ID"]
                                .reset_index(drop=True)
                            )
                        )

                        # Prepare original feature values for comparison
                        original_features = (
                            customer_data[required_features]
                            .apply(pd.to_numeric, errors="coerce")
                            .replace([np.inf, -np.inf], np.nan)
                            .reset_index(drop=True)
                        )

                        uploaded_features = input_features.reset_index(drop=True)

                        # Compare all feature values, treating NaN values as equal
                        same_feature_values = np.array_equal(
                            uploaded_features.to_numpy(dtype=float),
                            original_features.to_numpy(dtype=float),
                            equal_nan=True
                        )

                        # Confirm original data only when both checks pass
                        is_original_dataset = (
                            same_customer_ids and same_feature_values
                        )

                    if is_original_dataset:

                        original_labels = customer_data["Cluster"].to_numpy()

                        matching = np.sum(predictions == original_labels)

                        accuracy = matching / len(predictions) * 100

                        st.subheader("Prediction Validation")

                        st.metric(
                            "Matching Assignments",
                            f"{matching:,} / {len(predictions):,}"
                        )

                        st.metric(
                            "Exact Match Percentage",
                            f"{accuracy:.2f}%"
                        )

                        print("\nPREDICTION VALIDATION")
                        print("-" * 35)
                        print(f"Matching assignments: {matching:,}")
                        print(f"Total customers: {len(predictions):,}")
                        print(f"Exact match percentage: {accuracy:.2f}%")

                        if accuracy == 100:
                            st.success("All customer assignments match the original model results!")
                            print("Validation passed!")

                        else:
                            st.warning("Some customer assignments differ.")
                            print("Validation requires investigation.")

                    else:
                        st.info(
                            "New dataset detected. Predictions are generated "
                            "using the existing trained model. Original training "
                            "assignments are not available for comparison."
                        )

                        print("\nNew dataset detected.")
                        print("Training-data comparison skipped.")
                        print("Predictions generated successfully.")

                    print("=" * 45)
                    
                    # Create prediction results
                    prediction_results = uploaded_df.copy()

                    if "CUST_ID" not in prediction_results.columns:

                        prediction_results.insert(
                            0,
                            "CUST_ID",
                            [
                                f"Uploaded_Customer_{i+1}"
                                for i in range(len(prediction_results))
                            ]
                        )

                    prediction_results["Predicted_Cluster"] = predictions

                    # Display results
                    st.success(
                        f"Successfully segmented "
                        f"{len(prediction_results):,} customers!"
                    )

                    st.subheader("Prediction Summary")

                    prediction_counts = (
                        prediction_results["Predicted_Cluster"]
                        .value_counts()
                        .sort_index()
                        .to_frame("Customer Count")
                    )

                    prediction_counts["Percentage (%)"] = (
                        prediction_counts["Customer Count"]
                        / len(prediction_results) * 100
                    ).round(2)

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Customers Processed",
                        f"{len(prediction_results):,}"
                    )

                    col2.metric(
                        "Segments Identified",
                        prediction_results[
                            "Predicted_Cluster"
                        ].nunique()
                    )

                    st.bar_chart(
                        prediction_counts["Customer Count"]
                    )

                    st.dataframe(
                        prediction_counts,
                        width="stretch"
                    )

                    st.subheader("Customer Prediction Results")

                    st.dataframe(
                        prediction_results,
                        width="stretch",
                        hide_index=True
                    )

                    # Download predictions
                    csv_data = prediction_results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Customer Predictions",
                        data=csv_data,
                        file_name="customer_segmentation_predictions.csv",
                        mime="text/csv"
                    )

    except Exception as e:

        st.error(f"Unable to process the uploaded dataset: {e}")



# --------------------------------------------------
# 9. METHOD 2 - TRAIN A NEW MODEL ON AN UPLOADED DATASET
# --------------------------------------------------

st.divider()

st.header("Method 2: Discover New Customer Segments")
st.write(
    "Upload a compatible customer dataset to fit a new preprocessing "
    "pipeline and train a fresh K-Means model. This method discovers "
    "clusters within the uploaded dataset; it does not reuse the five "
    "cluster assignments from Method 1."
)

st.info(
    "The uploaded file must contain the same 16 customer-behaviour "
    "features used by this project. CUST_ID is optional and is not "
    "used for clustering. PURCHASES is not a clustering feature."
)

new_dataset_file = st.file_uploader(
    "Upload CSV for new clustering",
    type=["csv"],
    key="new_dataset_clustering_upload"
)

if new_dataset_file is not None:
    try:
        new_uploaded_df = pd.read_csv(new_dataset_file)
        upload_signature = hashlib.sha256(new_dataset_file.getvalue()).hexdigest()

        # Remove an old result if the user has uploaded a different file.
        previous_result = st.session_state.get("method2_result")
        if (
            previous_result is not None
            and previous_result.get("upload_signature") != upload_signature
        ):
            st.session_state.pop("method2_result", None)

        st.subheader("New Dataset Preview")
        st.caption(
            f"Rows: {len(new_uploaded_df):,} | "
            f"Columns: {len(new_uploaded_df.columns):,}"
        )
        st.dataframe(new_uploaded_df.head(), width="stretch")

        method2_required_features = model_bundle["selected_features"]
        method2_log_features = model_bundle["preprocessing_artifacts"]["log_features"]

        missing_method2_features = [
            feature for feature in method2_required_features
            if feature not in new_uploaded_df.columns
        ]

        if new_uploaded_df.empty:
            st.error("The uploaded dataset contains no customer records.")

        elif missing_method2_features:
            st.error(
                "The uploaded dataset is missing required features: "
                + ", ".join(missing_method2_features)
            )

        else:
            # Keep only the 16 model features and preserve training order.
            method2_input = new_uploaded_df[method2_required_features].copy()
            method2_input = method2_input.apply(pd.to_numeric, errors="coerce")
            method2_input = method2_input.replace([np.inf, -np.inf], np.nan)

            # Completely empty columns cannot be median-imputed.
            all_missing_features = method2_input.columns[
                method2_input.isna().all()
            ].tolist()

            negative_log_features = [
                feature for feature in method2_log_features
                if (method2_input[feature].dropna() < 0).any()
            ]

            if all_missing_features:
                st.error(
                    "These required features contain no usable numeric values: "
                    + ", ".join(all_missing_features)
                )

            elif negative_log_features:
                st.error(
                    "Negative values cannot be processed by log1p for these "
                    "features: " + ", ".join(negative_log_features)
                )

            elif len(method2_input) < 3:
                st.error("At least three customer records are required for clustering.")

            else:
                # METHOD 2 PREPROCESSING: fit new transformers on this upload.
                # These are not the imputer/scaler learned in Method 1.
                method2_imputer = SimpleImputer(strategy="median")
                method2_imputed_array = method2_imputer.fit_transform(method2_input)

                method2_imputed_df = pd.DataFrame(
                    method2_imputed_array,
                    columns=method2_required_features,
                    index=method2_input.index
                )

                # Apply the project's established log1p feature choices.
                method2_transformed_df = method2_imputed_df.copy()
                method2_transformed_df[method2_log_features] = np.log1p(
                    method2_transformed_df[method2_log_features]
                )

                # Fit a fresh scaler using only the uploaded dataset.
                method2_scaler = StandardScaler()
                method2_scaled_array = method2_scaler.fit_transform(
                    method2_transformed_df[method2_required_features]
                )
                method2_scaled_df = pd.DataFrame(
                    method2_scaled_array,
                    columns=method2_required_features,
                    index=method2_input.index
                )

                # Determine how many distinct customer profiles are available.
                unique_profiles = method2_imputed_df.round(10).drop_duplicates().shape[0]
                max_candidate_k = min(10, len(method2_scaled_df) - 1, unique_profiles)

                if max_candidate_k < 2:
                    st.error(
                        "The uploaded data contains too few distinct customer "
                        "profiles to evaluate K-Means clustering."
                    )

                else:
                    st.subheader("Step 1: Evaluate the Number of Clusters")
                    st.write(
                        "The app evaluates K values from 2 to "
                        f"{max_candidate_k}. Inertia measures within-cluster "
                        "sum of squared distances; the silhouette score measures "
                        "how well-separated the clusters are."
                    )

                    with st.spinner("Evaluating candidate cluster counts..."):
                        method2_metrics = evaluate_new_dataset_clusters(
                            method2_scaled_df,
                            max_candidate_k
                        )

                    valid_metrics = method2_metrics.dropna(
                        subset=["Silhouette Score"]
                    )

                    if valid_metrics.empty:
                        st.error(
                            "A valid silhouette score could not be calculated. "
                            "Check whether the uploaded records have enough "
                            "variation for clustering."
                        )

                    else:
                        best_silhouette_row = valid_metrics.loc[
                            valid_metrics["Silhouette Score"].idxmax()
                        ]
                        recommended_k = int(best_silhouette_row["K"])

                        st.dataframe(
                            method2_metrics.round(4),
                            width="stretch",
                            hide_index=True
                        )

                        inertia_col, silhouette_col = st.columns(2)
                        with inertia_col:
                            st.caption("Inertia by K (look for a bend/elbow)")
                            st.line_chart(
                                method2_metrics.set_index("K")["Inertia"]
                            )

                        with silhouette_col:
                            st.caption("Silhouette score by K (higher is better)")
                            st.line_chart(
                                method2_metrics.set_index("K")["Silhouette Score"]
                            )

                        st.success(
                            f"Highest tested silhouette score: K = {recommended_k} "
                            f"({best_silhouette_row['Silhouette Score']:.4f}). "
                            "Treat this as a recommendation, not an automatic proof "
                            "that this is the only correct cluster count."
                        )

                        candidate_k_values = [
                            int(value) for value in valid_metrics["K"].tolist()
                        ]
                        chosen_k = st.selectbox(
                            "Choose the number of clusters for the new model",
                            options=candidate_k_values,
                            index=candidate_k_values.index(recommended_k),
                            key="method2_selected_k"
                        )

                        st.caption(
                            "You can select the recommended K or choose another "
                            "value after reviewing the inertia and silhouette results."
                        )

                        if st.button(
                            "Train New K-Means Model",
                            type="primary",
                            key="train_new_kmeans_model"
                        ):
                            with st.spinner(f"Training a new K-Means model with K={chosen_k}..."):
                                new_kmeans_model = KMeans(
                                    n_clusters=int(chosen_k),
                                    init="k-means++",
                                    n_init=10,
                                    max_iter=300,
                                    random_state=42
                                )
                                new_cluster_labels = new_kmeans_model.fit_predict(
                                    method2_scaled_df
                                )

                                # Add new cluster IDs to a copy of the original upload.
                                method2_results = new_uploaded_df.copy()
                                if "CUST_ID" not in method2_results.columns:
                                    method2_results.insert(
                                        0,
                                        "CUST_ID",
                                        [
                                            f"Uploaded_Customer_{i + 1}"
                                            for i in range(len(method2_results))
                                        ]
                                    )
                                method2_results["New_Cluster"] = new_cluster_labels

                                # Build a readable profile in the original feature units.
                                method2_profile_source = method2_imputed_df.copy()
                                method2_profile_source["New_Cluster"] = new_cluster_labels
                                method2_cluster_profile = (
                                    method2_profile_source
                                    .groupby("New_Cluster")[method2_required_features]
                                    .median()
                                    .round(2)
                                )

                                st.session_state["method2_result"] = {
                                    "upload_signature": upload_signature,
                                    "results": method2_results,
                                    "profile": method2_cluster_profile,
                                    "selected_k": int(chosen_k),
                                    "actual_clusters": int(len(np.unique(new_cluster_labels))),
                                    "metrics": method2_metrics
                                }

                        # Render the latest trained result after the button click.
                        method2_saved_result = st.session_state.get("method2_result")
                        if (
                            method2_saved_result is not None
                            and method2_saved_result.get("upload_signature") == upload_signature
                        ):
                            st.divider()
                            st.subheader("Step 2: New Dataset Clustering Results")

                            saved_results = method2_saved_result["results"]
                            saved_profile = method2_saved_result["profile"]
                            saved_k = method2_saved_result["selected_k"]

                            st.success(
                                f"New K-Means model trained successfully with K={saved_k}. "
                                f"{len(saved_results):,} customers have been assigned to clusters."
                            )

                            st.warning(
                                "These cluster IDs belong to this newly trained model. "
                                "For example, New_Cluster 0 is not necessarily the same "
                                "customer segment as Cluster 0 in Method 1."
                            )

                            new_distribution = (
                                saved_results["New_Cluster"]
                                .value_counts()
                                .sort_index()
                                .rename_axis("New_Cluster")
                                .to_frame("Customer Count")
                            )
                            new_distribution["Percentage (%)"] = (
                                new_distribution["Customer Count"]
                                / len(saved_results) * 100
                            ).round(2)

                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            metric_col1.metric("Customers clustered", f"{len(saved_results):,}")
                            metric_col2.metric("K selected", saved_k)
                            metric_col3.metric(
                                "Clusters produced",
                                method2_saved_result["actual_clusters"]
                            )

                            st.caption("New cluster distribution")
                            st.bar_chart(new_distribution["Customer Count"])
                            st.dataframe(new_distribution, width="stretch")

                            st.subheader("New Cluster Behaviour Profile")
                            st.write(
                                "Median feature values for each new cluster, shown in "
                                "the original feature units after median imputation. "
                                "Use these profiles to interpret and name the segments."
                            )
                            st.dataframe(saved_profile, width="stretch")

                            st.subheader("Customer Assignments")
                            st.dataframe(saved_results, width="stretch", hide_index=True)

                            method2_csv = saved_results.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Download New Clustering Results",
                                data=method2_csv,
                                file_name="new_dataset_customer_clusters.csv",
                                mime="text/csv",
                                key="download_method2_results"
                            )

    except Exception as error:
        st.error(f"Unable to process the new clustering dataset: {error}")

# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Powered by Python, Pandas, Scikit-learn and Streamlit."
)