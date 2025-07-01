import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set page configuration
st.set_page_config(page_title="ML Models: Cardiovascular Disease", layout="wide")

# Title and description
st.title("Machine Learning on Cardiovascular Disease Dataset")
st.write("Upload the Cardiovascular Disease Dataset (CSV) and select a model to analyze the data.")

# File uploader
uploaded_file = st.file_uploader("Choose the Cardiovascular Disease Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Step 1: Load the dataset
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())
        st.write(f"Dataset Shape: {df.shape}")

        # Step 2: Preprocessing
        # Replace zeros in 'serumcholestrol' with median
        df['serumcholestrol'] = df['serumcholestrol'].replace(0, df['serumcholestrol'].median())

        # Encode categorical features
        categorical_cols = ['gender', 'chestpain', 'restingrelectro', 'slope']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Standardize numerical features
        numerical_cols = ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak', 'noofmajorvessels', 'fastingbloodsugar', 'exerciseangia']
        scaler = StandardScaler()
        df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

        # Create X and y
        X = df_encoded.drop(['patientid', 'target'], axis=1)
        y = df['target']
        st.write(f"### Preprocessed Data Shape: {X.shape}")

        # Model selection
        st.write("### Select Model")
        model_choice = st.selectbox("Choose a model to run:", 
                                    ["Select a model", "K-means Clustering", "Logistic Regression", "Random Forest Classifier", "Compare LR and RFC"],
                                    index=0)

        if model_choice == "Select a model":
            st.info("Please select a model from the dropdown to proceed with analysis.")
        else:
            if model_choice == "K-means Clustering":
                # Step 3: K-means clustering to find optimal K
                st.write("### Select Number of Clusters")
                n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=4)

                inertias = []
                sil_scores = []
                K_range = range(2, 11)
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(X)
                    inertias.append(kmeans.inertia_)
                    sil_scores.append(silhouette_score(X, kmeans.labels_))

                # Step 4: Plot elbow curve
                st.write("### Elbow Method for Optimal K")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(K_range, inertias, 'bo-')
                ax.set_xlabel('Number of Clusters (K)')
                ax.set_ylabel('Inertia')
                ax.set_title('Elbow Method for Optimal K')
                plt.tight_layout()
                st.pyplot(fig)

                # Step 5: Plot silhouette scores
                st.write("### Silhouette Score vs. K")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(K_range, sil_scores, 'ro-')
                ax.set_xlabel('Number of Clusters (K)')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Silhouette Score vs. K')
                plt.tight_layout()
                st.pyplot(fig)

                # Step 6: Apply K-means clustering with selected K
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df['cluster'] = kmeans.fit_predict(X)
                st.write(f"### K-means Clustering Completed with K={n_clusters}")
                st.write("Clustered Data Preview:")
                st.write(df.head())

                # Step 7: Visualize age distribution across clusters
                st.write("### Age Distribution Across Clusters")
                cluster_ages = [df[df['cluster'] == i]['age'].tolist() for i in range(n_clusters)]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.boxplot(cluster_ages, labels=[f'Cluster {i+1}' for i in range(n_clusters)])
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Age')
                ax.set_title('Age Distribution Across K-means Clusters')
                plt.tight_layout()
                st.pyplot(fig)

                # Step 8: PCA for 2D visualization
                st.write("### K-means Clusters in 2D PCA Space")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                fig, ax = plt.subplots(figsize=(8, 5))
                for i in range(n_clusters):
                    ax.scatter(X_pca[df['cluster'] == i, 0], X_pca[df['cluster'] == i, 1], label=f'Cluster {i+1}')
                ax.set_xlabel('PCA Component 1')
                ax.set_ylabel('PCA Component 2')
                ax.set_title('K-means Clusters in 2D PCA Space')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                # Step 9: Silhouette score for selected K
                sil_score = silhouette_score(X, df['cluster'])
                st.write(f"### Silhouette Score for K={n_clusters}: {sil_score:.3f}")

                # Step 10: Summarize clusters
                st.write("### Cluster Summary")
                numerical_summary = df.groupby('cluster')[numerical_cols].mean()
                categorical_summary = df.groupby('cluster')[['gender', 'chestpain', 'restingrelectro', 'slope']].agg(lambda x: x.mode()[0])
                cluster_summary = pd.concat([numerical_summary, categorical_summary], axis=1)
                st.write(cluster_summary)

            else:
                # Train/test split for supervised models
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                if model_choice == "Logistic Regression":
                    # Train Logistic Regression
                    lr = LogisticRegression(random_state=42, max_iter=1000)
                    lr.fit(X_train_scaled, y_train)
                    y_pred = lr.predict(X_test_scaled)
                    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
                    df['lr_prediction'] = lr.predict(X)

                    # Compute metrics
                    metrics = {
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1-Score': f1_score(y_test, y_pred),
                        'ROC AUC': roc_auc_score(y_test, y_pred_proba)
                    }
                    st.write("### Logistic Regression Results")
                    st.write(pd.DataFrame(metrics, index=["Logistic Regression"]))

                    # Plot 1: Confusion Matrix
                    st.write("### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
                    ax.set_title('Confusion Matrix for Logistic Regression')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Plot 2: ROC Curve
                    st.write("### ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["ROC AUC"]:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve for Logistic Regression')
                    ax.legend(loc='lower right')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Plot 3: Feature Importance
                    st.write("### Feature Importance")
                    feature_names = X.columns
                    coefficients = lr.coef_[0]
                    feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
                    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
                    ax.set_title('Feature Importance in Logistic Regression')
                    ax.set_xlabel('Coefficient Value')
                    ax.set_ylabel('Feature')
                    plt.tight_layout()
                    st.pyplot(fig)

                elif model_choice == "Random Forest Classifier":
                    # Train Random Forest
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_train, y_train)  # Unscaled data
                    y_pred = rf.predict(X_test)
                    y_pred_proba = rf.predict_proba(X_test)[:, 1]
                    df['rf_prediction'] = rf.predict(X)

                    # Compute metrics
                    metrics = {
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1-Score': f1_score(y_test, y_pred),
                        'ROC AUC': roc_auc_score(y_test, y_pred_proba)
                    }
                    st.write("### Random Forest Classifier Results")
                    st.write(pd.DataFrame(metrics, index=["Random Forest Classifier"]))

                    # Cross-validation
                    cv_roc_auc = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
                    st.write(f"### Cross-Validated ROC AUC")
                    st.write(f"Scores: {cv_roc_auc}")
                    st.write(f"Mean: {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std() * 2:.4f})")

                    # Plot 1: Confusion Matrix
                    st.write("### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
                    ax.set_title('Confusion Matrix for Random Forest')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Plot 2: ROC Curve
                    st.write("### ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["ROC AUC"]:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve for Random Forest')
                    ax.legend(loc='lower right')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Plot 3: Feature Importance
                    st.write("### Feature Importance")
                    feature_names = X.columns
                    importances = rf.feature_importances_
                    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance)
                    ax.set_title('Feature Importance in Random Forest')
                    ax.set_xlabel('Importance Score')
                    ax.set_ylabel('Feature')
                    plt.tight_layout()
                    st.pyplot(fig)

                elif model_choice == "Compare LR and RFC":
                    # Train both models
                    lr = LogisticRegression(random_state=42, max_iter=1000)
                    lr.fit(X_train_scaled, y_train)
                    lr_y_pred = lr.predict(X_test_scaled)
                    lr_y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
                    df['lr_prediction'] = lr.predict(X)

                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_train, y_train)
                    rf_y_pred = rf.predict(X_test)
                    rf_y_pred_proba = rf.predict_proba(X_test)[:, 1]
                    df['rf_prediction'] = rf.predict(X)

                    # Compute metrics
                    lr_metrics = {
                        'Accuracy': accuracy_score(y_test, lr_y_pred),
                        'Precision': precision_score(y_test, lr_y_pred),
                        'Recall': recall_score(y_test, lr_y_pred),
                        'F1-Score': f1_score(y_test, lr_y_pred),
                        'ROC AUC': roc_auc_score(y_test, lr_y_pred_proba)
                    }
                    rf_metrics = {
                        'Accuracy': accuracy_score(y_test, rf_y_pred),
                        'Precision': precision_score(y_test, rf_y_pred),
                        'Recall': recall_score(y_test, rf_y_pred),
                        'F1-Score': f1_score(y_test, rf_y_pred),
                        'ROC AUC': roc_auc_score(y_test, rf_y_pred_proba)
                    }
                    metrics_df = pd.DataFrame([lr_metrics, rf_metrics], index=["Logistic Regression", "Random Forest Classifier"])
                    st.write("### Model Comparison Metrics")
                    st.write(metrics_df)

                    # Plot bar chart
                    st.write("### Performance Comparison")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.arange(len(metrics_df.columns))
                    width = 0.35
                    rects1 = ax.bar(x - width/2, metrics_df.loc["Random Forest Classifier"], width, label='Random Forest Classifier', color='skyblue')
                    rects2 = ax.bar(x + width/2, metrics_df.loc["Logistic Regression"], width, label='Logistic Regression', color='lightcoral')
                    ax.set_ylabel('Scores')
                    ax.set_title('Performance Comparison: Random Forest Classifier vs Logistic Regression')
                    ax.set_xticks(x)
                    ax.set_xticklabels(metrics_df.columns)
                    ax.legend()

                    def autolabel(rects):
                        for rect in rects:
                            height = rect.get_height()
                            ax.annotate(f'{height:.4f}',
                                        xy=(rect.get_x() + rect.get_width() / 2, height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom')
                    autolabel(rects1)
                    autolabel(rects2)
                    plt.tight_layout()
                    st.pyplot(fig)

            # Download option for dataset with predictions
            st.write("### Download Dataset with Predictions")
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            st.download_button(
                label="Download CSV with Cluster Labels and Predictions",
                data=buffer.getvalue(),
                file_name="Cardiovascular_Disease_Dataset_with_Results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload the Cardiovascular Disease Dataset CSV file to proceed.")

# Instructions for running the app
st.write("""
### Instructions
1. Upload the Cardiovascular Disease Dataset CSV file.
2. Select a model from the dropdown (K-means, Logistic Regression, Random Forest, or Compare LR and RFC).
3. For K-means: Choose the number of clusters and view clustering results.
4. For Logistic Regression/Random Forest: View performance metrics, confusion matrix, ROC curve, and feature importance.
5. For Compare LR and RFC: View a bar chart comparing performance metrics.
6. Download the dataset with cluster labels and model predictions.
""")