import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

def detect_data_type(df: pd.DataFrame) -> str:
    """Detect the type of data in the DataFrame"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    binary_cols = []
    categorical_cols = []
    
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        
        # Check for binary data (exactly 2 unique values)
        if len(unique_vals) == 2:
            binary_cols.append(col)
        # Check for categorical data (non-numeric with limited unique values)
        elif not pd.api.types.is_numeric_dtype(df[col]) and len(unique_vals) < 20:
            categorical_cols.append(col)
    
    # Determine data type
    if len(numeric_cols) == len(df.columns):
        return "Numerical"
    elif len(binary_cols) == len(df.columns):
        return "Binary"
    elif len(categorical_cols) == len(df.columns):
        return "Categorical"
    elif all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        return "Ordinal" if df.nunique().mean() < 10 else "Numerical"
    else:
        return "Mixed"

def get_distance_metrics_for_datatype(data_type: str) -> dict:
    """Return available distance metrics for the given data type"""
    metrics = {
        "Numerical": {
            "Euclidean": "Standard straight-line distance between points",
            "Manhattan": "Sum of absolute differences between coordinates",
            "Minkowski": "Generalization of Euclidean and Manhattan distances",
            "Cosine Distance": "1 minus the cosine of the angle between points",
            "Correlation (Pearson)": "1 minus the Pearson correlation coefficient",
            "Mahalanobis": "Distance relative to the data distribution",
            "Chebyshev": "Maximum coordinate difference between points",
            "Bray-Curtis": "Difference divided by sum for each dimension",
            "Canberra": "Weighted version of Manhattan distance",
            "Squared Euclidean": "Square of standard Euclidean distance"
        },
        "Binary": {
            "Hamming Distance": "Proportion of differing bits",
            "Jaccard Distance": "1 minus the Jaccard similarity coefficient",
            "Simple Matching (SMC)": "Proportion of matching attributes",
            "Dice Coefficient": "Similarity measure for binary data",
            "Russell-Rao": "Proportion of attributes present in both items",
            "Yule's Q": "Measure of association between binary variables",
            "Tanimoto": "Extended version of Jaccard for weighted features",
            "Kulczynski": "Ratio of positive matches to non-matches",
            "Ochiai": "Cosine similarity for binary data",
            "Sokal-Michener": "Simple matching coefficient extended"
        },
        "Categorical": {
            "Overlap Metric": "Count of matching categories",
            "Hamming": "Proportion of mismatching categories",
            "Goodall's Measure": "Rare matches are more significant",
            "Lin's Measure": "Information-theoretic similarity",
            "Inverse Frequency": "Weight by inverse category frequency",
            "Jaccard for Categories": "Jaccard similarity for categories",
            "Mutual Information": "Information shared between variables",
            "Ng's Measure": "Weighted by marginal probabilities",
            "Entropy-based": "Based on information gain",
            "VDM (Value Difference Metric)": "Probability-based distance"
        },
        "Ordinal": {
            "Rank-based Euclidean": "Euclidean on rank-transformed data",
            "Spearman Distance": "1 minus Spearman correlation",
            "Ordinal Manhattan": "Manhattan on rank-transformed data",
            "Gower for Ordinal": "Gower's similarity for ordinal data",
            "Cumulative Overlap": "Based on cumulative distribution",
            "Ordinal VDM": "Value Difference Metric for ordinal",
            "Weighted Kappa": "Agreement measure for ordinal data",
            "Ordinal Cosine": "Cosine similarity on ranks",
            "Borda Count Distance": "Based on Borda count aggregation",
            "Kendall Tau Distance": "Based on Kendall's rank correlation"
        },
        "Mixed": {
            "Gower Distance": "Handles mixed data types",
            "HEOM": "Heterogeneous Euclidean-Overlap Metric",
            "HVDM": "Heterogeneous Value Difference Metric",
            "Generalized Mahalanobis": "Mahalanobis for mixed data",
            "Distance Fusion": "Combines multiple distance metrics",
            "Hybrid Similarity": "Custom hybrid similarity measure",
            "K-Prototypes Distance": "For clustering mixed data",
            "Mixed-Attribute Dissimilarity": "Combined dissimilarity",
            "Feature Hashing Distance": "Using feature hashing",
            "Meta-Learning Distance": "Learns optimal distance metric"
        }
    }
    
    return metrics.get(data_type, {})

def get_supported_metrics_for_distance(distance_metric: str) -> dict:
    """Return supported evaluation metrics for a given distance metric"""
    metrics = {
        "silhouette": "Measures how similar an object is to its own cluster compared to other clusters",
        "davies_bouldin": "Ratio of within-cluster distances to between-cluster distances",
        "calinski_harabasz": "Ratio of between-cluster dispersion to within-cluster dispersion"
    }
    
    return metrics

def preprocess_data(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Preprocess data based on its type"""
    if data_type == "Numerical":
        # Standardize numerical data
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    elif data_type == "Binary":
        # Ensure binary data is encoded as 0/1
        return df.apply(lambda x: x.astype(int) if x.nunique() == 2 else pd.get_dummies(x, drop_first=True))
    
    elif data_type == "Categorical":
        # One-hot encode categorical data
        return pd.get_dummies(df)
    
    elif data_type == "Ordinal":
        # Label encode ordinal data
        le = LabelEncoder()
        return df.apply(lambda x: le.fit_transform(x) if not pd.api.types.is_numeric_dtype(x) else x)
    
    else:  # Mixed data
        # Handle each column according to its type
        processed_data = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                scaler = StandardScaler()
                processed_col = scaler.fit_transform(df[[col]])
            else:
                # For categorical data in mixed dataset
                processed_col = pd.get_dummies(df[col], prefix=col)
            
            processed_data.append(processed_col if isinstance(processed_col, pd.DataFrame) 
                                else pd.DataFrame(processed_col, columns=[col]))
        
        return pd.concat(processed_data, axis=1)

def find_optimal_k(processed_df, k_range=(2, 10)):
    """Find optimal k using silhouette analysis"""
    silhouette_scores = []
    
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(processed_df)
        silhouette_avg = silhouette_score(processed_df, labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_k = np.argmax(silhouette_scores) + k_range[0]
    return optimal_k

def perform_clustering(df: pd.DataFrame, n_clusters: int, 
                      distance_metric: str, data_type: str) -> dict:
    """Perform K-means clustering with the specified parameters"""
    # Preprocess data
    processed_df = preprocess_data(df, data_type)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(processed_df)
    
    # Calculate evaluation metrics
    silhouette = silhouette_score(processed_df, clusters)
    davies_bouldin = davies_bouldin_score(processed_df, clusters)
    calinski_harabasz = calinski_harabasz_score(processed_df, clusters)
    
    # Add cluster labels to original data
    clustered_data = df.copy()
    clustered_data['Cluster'] = clusters
    
    # Get cluster counts
    cluster_counts = clustered_data['Cluster'].value_counts().sort_index()
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'cluster_counts': cluster_counts,
        'clustered_data': clustered_data,
        'model': kmeans
    }