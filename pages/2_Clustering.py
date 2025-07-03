import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from utils.clustering import (
    detect_data_type,
    get_distance_metrics_for_datatype,
    perform_clustering,
    get_supported_metrics_for_distance,
    find_optimal_k,
    preprocess_data
)
import os
from sklearn.metrics import silhouette_samples
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Page configuration
st.set_page_config(page_title="Clustering Analysis", layout="wide")
st.title("📈 K-Means Clustering Analysis")

def display_data_analysis(df):
    """Display detailed data analysis"""
    st.subheader("🔍 Data Analysis Results")
    
    # Detect data type
    data_type = detect_data_type(df)
    st.session_state['data_type'] = data_type
    
    # Display data type information
    type_cols = st.columns(3)
    type_cols[0].metric("Data Type", data_type)
    
    # Display column information
    st.write("**Column Information:**")
    col_info = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_vals = df[col].nunique()
        null_count = df[col].isnull().sum()
        col_info.append({
            "Column": col,
            "Data Type": dtype,
            "Unique Values": unique_vals,
            "Null Values": null_count
        })
    
    st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    
    # Show distance metrics based on data type
    st.subheader("📏 Available Distance Metrics")
    distance_metrics = get_distance_metrics_for_datatype(data_type)
    
    st.info(f"Based on your {data_type} data, the following distance metrics are available:")
    st.write(distance_metrics)
    
    # Store distance metrics in session state
    st.session_state['distance_metrics'] = distance_metrics
    
    return data_type

def plot_elbow_method(processed_df, k_range):
    """Plot elbow method results"""
    st.subheader("📉 Elbow Method Analysis")
    
    model = KMeans(random_state=42)
    visualizer = KElbowVisualizer(model, k=k_range, timings=False)
    visualizer.fit(processed_df)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add elbow curve
    fig.add_trace(go.Scatter(
        x=visualizer.k_values_,
        y=visualizer.k_scores_,
        mode='lines+markers',
        name='Distortion Score',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add optimal K marker
    if visualizer.elbow_value_ is not None:
        fig.add_vline(
            x=visualizer.elbow_value_,
            line=dict(color='red', dash='dash'),
            annotation_text=f"Optimal K: {visualizer.elbow_value_}",
            annotation_position="top right"
        )
    
    fig.update_layout(
        title='Elbow Method For Optimal k',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Distortion Score',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return visualizer.elbow_value_

def plot_silhouette_analysis(processed_df, optimal_k):
    """Plot silhouette analysis results"""
    st.subheader("📊 Silhouette Analysis")
    
    visualizer = SilhouetteVisualizer(KMeans(optimal_k, random_state=42), colors='yellowbrick')
    visualizer.fit(processed_df)
    
    # Create Plotly figure from the visualizer data
    fig = go.Figure()
    
    # Add silhouette samples
    y_lower = 10
    for i in range(optimal_k):
        # Aggregate the silhouette scores for samples belonging to cluster i
        ith_cluster_silhouette_values = visualizer.silhouette_samples_[visualizer.labels_ == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        # Add cluster silhouette
        fig.add_trace(go.Scatter(
            x=ith_cluster_silhouette_values,
            y=np.arange(y_lower, y_upper),
            mode='lines',
            line=dict(width=1),
            fill='tozerox',
            name=f'Cluster {i}',
            hoverinfo='x+name'
        ))
        
        y_lower = y_upper + 10
    
    # Add average line
    fig.add_vline(
        x=visualizer.silhouette_score_,
        line=dict(color='red', dash='dash'),
        annotation_text=f"Avg Silhouette: {visualizer.silhouette_score_:.2f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='Silhouette Plot for KMeans Clustering',
        xaxis_title='Silhouette Coefficient Values',
        yaxis_title='Cluster',
        showlegend=True,
        yaxis=dict(showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return visualizer.silhouette_score_

def plot_animated_clusters(df, clusters, n_clusters):
    """Create animated cluster visualization"""
    st.subheader("🎬 Animated Cluster Visualization")
    
    # Select two most important features (for visualization)
    if len(df.columns) >= 2:
        # For simplicity, use first two numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
        else:
            x_col, y_col = df.columns[0], df.columns[1]
        
        # Create animated scatter plot
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=clusters.astype(str),
            animation_frame=df.index % 10,  # Simple animation frame
            title="Animated Cluster Visualization",
            labels={"color": "Cluster"},
            hover_data=df.columns,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Add cluster centers if available
        if 'model' in st.session_state.get('clustering_results', {}):
            centers = st.session_state['clustering_results']['model'].cluster_centers_
            if len(centers[0]) >= 2:
                fig.add_trace(go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode='markers',
                    marker=dict(
                        color='black',
                        size=10,
                        symbol='x',
                        line=dict(width=2)
                    ),
                    name='Cluster Centers'
                ))
        
        fig.update_layout(
            transition={'duration': 500},
            updatemenus=[{
                'buttons': [{
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 
                                    'fromcurrent': True, 'transition': {'duration': 300}}],
                    'label': 'Play',
                    'method': 'animate'
                }, {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 
                                      'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least 2 dimensions for visualization")

def recommend_distance_metrics(data_type, df):
    """Recommend top 5 distance metrics for the data with enhanced UI"""
    with st.container():
        st.subheader("🏆 Top 5 Recommended Distance Metrics")
        st.markdown("""
        <style>
        .metric-card {
            padding: 15px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin-bottom: 10px;
            border-left: 4px solid #4e79a7;
        }
        .metric-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .metric-desc {
            font-size: 14px;
            color: #7f8c8d;
        }
        </style>
        """, unsafe_allow_html=True)
        
        recommendations = {
            "Numerical": [
                ("Euclidean", "Best for spherical clusters with similar variances", "⭐️⭐️⭐️⭐️⭐️"),
                ("Manhattan", "Robust to outliers, good for high-dimensional data", "⭐️⭐️⭐️⭐️"),
                ("Cosine", "Ideal for text data or when magnitude isn't important", "⭐️⭐️⭐️⭐️"),
                ("Correlation", "Good for data with linear relationships", "⭐️⭐️⭐️"),
                ("Mahalanobis", "Accounts for covariance between features", "⭐️⭐️⭐️⭐️")
            ],
            "Binary": [
                ("Jaccard", "Best for asymmetric binary attributes", "⭐️⭐️⭐️⭐️⭐️"),
                ("Hamming", "Simple count of mismatches", "⭐️⭐️⭐️⭐️"),
                ("SMC", "Good for symmetric binary data", "⭐️⭐️⭐️"),
                ("Dice", "Similar to Jaccard but weights matches more", "⭐️⭐️⭐️⭐️"),
                ("Yule's Q", "Good for measuring association", "⭐️⭐️⭐️")
            ],
            "Categorical": [
                ("Hamming", "Simple count of mismatches", "⭐️⭐️⭐️⭐️"),
                ("Jaccard", "Good for comparing sets of categories", "⭐️⭐️⭐️⭐️⭐️"),
                ("Goodall's", "Rare matches are more significant", "⭐️⭐️⭐️⭐️"),
                ("VDM", "Probability-based distance", "⭐️⭐️⭐️⭐️"),
                ("Inverse Frequency", "Weights rare categories more", "⭐️⭐️⭐️")
            ],
            "Ordinal": [
                ("Spearman", "Based on rank correlation", "⭐️⭐️⭐️⭐️⭐️"),
                ("Rank Euclidean", "Euclidean on rank-transformed data", "⭐️⭐️⭐️⭐️"),
                ("Gower", "Handles ordinal data well", "⭐️⭐️⭐️⭐️⭐️"),
                ("Weighted Kappa", "Accounts for ordering", "⭐️⭐️⭐️"),
                ("Kendall Tau", "Based on rank concordance", "⭐️⭐️⭐️⭐️")
            ],
            "Mixed": [
                ("Gower", "Best overall for mixed data", "⭐️⭐️⭐️⭐️⭐️"),
                ("HEOM", "Good for numerical + categorical", "⭐️⭐️⭐️⭐️"),
                ("HVDM", "Extension of VDM for mixed data", "⭐️⭐️⭐️⭐️"),
                ("Distance Fusion", "Combines multiple metrics", "⭐️⭐️⭐️"),
                ("K-Prototypes", "Specialized for clustering mixed data", "⭐️⭐️⭐️⭐️⭐️")
            ]
        }
        
        top_metrics = recommendations.get(data_type, [])
        
        if not top_metrics:
            st.warning("No specific recommendations for this data type")
            return
        
        for idx, (metric, desc, rating) in enumerate(top_metrics[:5]):
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">#{idx+1} {metric} <span style="float:right;">{rating}</span></div>
                <div class="metric-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption("💡 Ratings indicate general suitability for this data type (5 stars = most recommended)")
        
        return [m[0] for m in top_metrics[:5]]

def create_distance_metric_downloads(df, data_type, top_metrics):
    """Create enhanced download section for each recommended metric"""
    with st.container():
        st.subheader("📥 Download Cluster Results by Metric")
        st.markdown("""
        <style>
        .download-card {
            padding: 15px;
            border-radius: 10px;
            background-color: #f8f9fa;
            margin-bottom: 15px;
            border: 1px solid #dee2e6;
        }
        .download-header {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .download-desc {
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if not top_metrics:
            st.warning("No recommended metrics available for download")
            return
        
        # Store results in session state to prevent recomputation
        if 'download_results' not in st.session_state:
            st.session_state.download_results = {}
        
        for metric in top_metrics:
            with st.container():
                st.markdown(f"""
                <div class="download-card">
                    <div class="download-header">Distance Metric: {metric}</div>
                    <div class="download-desc">Download the clustered dataset using the {metric} distance metric</div>
                """, unsafe_allow_html=True)
                
                try:
                    # Check if we already have results for this metric
                    if metric not in st.session_state.download_results:
                        # Perform clustering with this metric
                        results = perform_clustering(
                            df,
                            n_clusters=st.session_state.get('n_clusters', 3),
                            distance_metric=metric,
                            data_type=data_type
                        )
                        st.session_state.download_results[metric] = results
                    else:
                        results = st.session_state.download_results[metric]
                    
                    # Create download button
                    csv = results['clustered_data'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download {metric} Results",
                        data=csv,
                        file_name=f'clustered_data_{metric}.csv',
                        mime='text/csv',
                        help=f"Clustered data using {metric} distance",
                        use_container_width=True,
                        key=f"download_{metric}"  # Unique key for each button
                    )
                    
                    # Show quick stats
                    cols = st.columns(3)
                    cols[0].metric("Silhouette", f"{results['silhouette']:.3f}")
                    cols[1].metric("DB Index", f"{results['davies_bouldin']:.3f}")
                    cols[2].metric("CH Index", f"{results['calinski_harabasz']:.3f}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error with {metric}: {str(e)}")

def display_supported_metrics(selected_metric):
    """Enhanced display of supported metrics for selected distance"""
    with st.container():
        st.subheader("📐 Supported Evaluation Metrics")
        st.markdown("""
        <style>
        .metric-table {
            width: 100%;
            border-collapse: collapse;
        }
        .metric-table th {
            background-color: #4e79a7;
            color: white;
            padding: 10px;
            text-align: left;
        }
        .metric-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .metric-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        </style>
        """, unsafe_allow_html=True)
        
        supported_metrics = {
            "silhouette": {
                "description": "Measures how similar an object is to its own cluster compared to other clusters",
                "range": "[-1, 1] (Higher is better)",
                "interpretation": "Values near +1 indicate well-clustered data"
            },
            "davies_bouldin": {
                "description": "Ratio of within-cluster distances to between-cluster distances",
                "range": "[0, ∞) (Lower is better)",
                "interpretation": "Values closer to 0 indicate better clustering"
            },
            "calinski_harabasz": {
                "description": "Ratio of between-cluster dispersion to within-cluster dispersion",
                "range": "[0, ∞) (Higher is better)",
                "interpretation": "Higher values indicate better defined clusters"
            }
        }
        
        # Create a styled table
        st.markdown(f"""
        <table class="metric-table">
            <tr>
                <th>Metric</th>
                <th>Description</th>
                <th>Range</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Silhouette Score</td>
                <td>{supported_metrics['silhouette']['description']}</td>
                <td>{supported_metrics['silhouette']['range']}</td>
                <td>{supported_metrics['silhouette']['interpretation']}</td>
            </tr>
            <tr>
                <td>Davies-Bouldin Index</td>
                <td>{supported_metrics['davies_bouldin']['description']}</td>
                <td>{supported_metrics['davies_bouldin']['range']}</td>
                <td>{supported_metrics['davies_bouldin']['interpretation']}</td>
            </tr>
            <tr>
                <td>Calinski-Harabasz Index</td>
                <td>{supported_metrics['calinski_harabasz']['description']}</td>
                <td>{supported_metrics['calinski_harabasz']['range']}</td>
                <td>{supported_metrics['calinski_harabasz']['interpretation']}</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #e8f4f8; border-radius: 5px;">
            <strong>💡 Pro Tip:</strong> When evaluating clustering results, consider multiple metrics together. 
            No single metric tells the complete story - look for consistent patterns across metrics.
        </div>
        """, unsafe_allow_html=True)

def display_clustering_results(results: dict, df: pd.DataFrame):
    """Display clustering results with visualizations and metrics"""
    
    # Print evaluation metrics
    print("\nClustering Evaluation Metrics:")
    print(f"Silhouette Score: {results['silhouette']:.3f}")
    print(f"Davies-Bouldin Index: {results['davies_bouldin']:.3f}")
    print(f"Calinski-Harabasz Index: {results['calinski_harabasz']:.3f}")
    
    # Display cluster counts
    print("\nCluster Sizes:")
    print(results['cluster_counts'])
    
    # Create interactive 3D scatter plot if there are at least 3 numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) >= 3:
        fig = go.Figure()
        
        for cluster in sorted(results['clustered_data']['Cluster'].unique()):
            cluster_data = results['clustered_data'][results['clustered_data']['Cluster'] == cluster]
            fig.add_trace(go.Scatter3d(
                x=cluster_data[numeric_cols[0]],
                y=cluster_data[numeric_cols[1]],
                z=cluster_data[numeric_cols[2]],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(size=5, opacity=0.7)
            ))
        
        fig.update_layout(
            title='3D Cluster Visualization',
            scene=dict(
                xaxis_title=numeric_cols[0],
                yaxis_title=numeric_cols[1],
                zaxis_title=numeric_cols[2]
            ),
            height=800
        )
        fig.show()
    elif len(numeric_cols) == 2:
        # Create 2D scatter plot if only 2 numeric columns
        fig = go.Figure()
        
        for cluster in sorted(results['clustered_data']['Cluster'].unique()):
            cluster_data = results['clustered_data'][results['clustered_data']['Cluster'] == cluster]
            fig.add_trace(go.Scatter(
                x=cluster_data[numeric_cols[0]],
                y=cluster_data[numeric_cols[1]],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(size=8, opacity=0.7)
            ))
        
        fig.update_layout(
            title='2D Cluster Visualization',
            xaxis_title=numeric_cols[0],
            yaxis_title=numeric_cols[1],
            height=600
        )
        fig.show()
    
    # Return the clustered data for further analysis
    return results['clustered_data']

def main():
    # Check if data is uploaded
    if 'uploaded_df' not in st.session_state:
        st.warning("Please upload data on the Data Upload page first.")
        st.page_link("pages/1_📊_Data_Upload.py", label="← Go to Data Upload", icon="⬅️")
        return
    
    df = st.session_state['uploaded_df']
    
    # Data analysis section
    if st.button("🔍 Analyze Dataset", help="Analyze the dataset to determine data type and available distance metrics"):
        data_type = display_data_analysis(df)
        st.session_state['analysis_done'] = True
        
        # Get and display recommended metrics
        top_metrics = recommend_distance_metrics(data_type, df)
        st.session_state['top_metrics'] = top_metrics
    
    # Clustering section (only shown after analysis)
    # In the main() function, modify the clustering section:

    if st.session_state.get('analysis_done', False):
        st.divider()
        st.subheader("⚙️ Clustering Configuration")
        
        # Preprocess data
        processed_df = preprocess_data(df, st.session_state['data_type'])
        
        # K selection method
        k_method = st.radio(
            "Select K Determination Method",
            ["Manual Selection", "Automatic (Elbow Method)", "Automatic (Silhouette Analysis)"],
            horizontal=True
        )
        
        selected_metric = st.selectbox(
            "Distance Metric",
            options=st.session_state.get('distance_metrics', {}).keys(),
            help="Select the distance metric to use for clustering"
        )
        
        if k_method == "Manual Selection":
            n_clusters = st.slider(
                "Number of Clusters (k)",
                min_value=2,
                max_value=min(10, len(df)),
                value=3,
                help="Select the number of clusters to create"
            )
        else:
            max_k = min(10, len(df))
            k_range = (2, max_k)
            
            if k_method == "Automatic (Elbow Method)":
                optimal_k = plot_elbow_method(processed_df, k_range)
            else:  # Silhouette Analysis
                optimal_k = find_optimal_k(processed_df, k_range)
                plot_silhouette_analysis(processed_df, optimal_k)
            
            if optimal_k is not None:
                st.success(f"Recommended number of clusters: {optimal_k}")
                n_clusters = st.number_input(
                    "Confirm number of clusters to use",
                    min_value=2,
                    max_value=max_k,
                    value=optimal_k
                )
            else:
                st.warning("Could not determine optimal k automatically. Please select manually.")
                n_clusters = st.slider(
                    "Number of Clusters (k)",
                    min_value=2,
                    max_value=max_k,
                    value=3
                )
        
        st.session_state['n_clusters'] = n_clusters
        
        # Check if we already have results for these parameters
        current_params = (selected_metric, n_clusters)
        if 'clustering_params' in st.session_state and st.session_state.clustering_params == current_params:
            # Use existing results
            results = st.session_state['clustering_results']
            display_clustering_results(results, df)
            plot_animated_clusters(df, results['clustered_data']['Cluster'], n_clusters)
            display_supported_metrics(selected_metric)
            create_distance_metric_downloads(df, st.session_state['data_type'], 
                                        st.session_state.get('top_metrics', []))
        else:
            if st.button("🚀 Perform Clustering", type="primary"):
                with st.spinner("Performing clustering... This may take some time for large datasets"):
                    try:
                        # Get supported metrics for the selected distance
                        supported_metrics = get_supported_metrics_for_distance(selected_metric)
                        
                        # Perform clustering
                        results = perform_clustering(
                            df,
                            n_clusters=n_clusters,
                            distance_metric=selected_metric,
                            data_type=st.session_state['data_type']
                        )
                        
                        # Display results
                        display_clustering_results(results, df)
                        plot_animated_clusters(df, results['clustered_data']['Cluster'], n_clusters)
                        
                        # Store results and parameters in session state
                        st.session_state['clustering_results'] = results
                        st.session_state['selected_metric'] = selected_metric
                        st.session_state['clustering_params'] = current_params
                        
                        # Display supported metrics
                        display_supported_metrics(selected_metric)
                        
                        # Create downloads for recommended metrics
                        create_distance_metric_downloads(df, st.session_state['data_type'], 
                                                    st.session_state.get('top_metrics', []))
                        
                    except Exception as e:
                        st.error(f"Error during clustering: {str(e)}")

if __name__ == "__main__":
    main()