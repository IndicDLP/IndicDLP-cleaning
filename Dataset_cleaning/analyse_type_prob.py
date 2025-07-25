import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv('/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/dmAP_all.csv', sep='\t')

# Calculate performance drops between IoU thresholds
for i in range(5, 9):
    df[f'mAP_drop_{i}_to_{i+1}'] = df[f'mAP_{i}'] - df[f'mAP_{i+1}']

# Calculate average mAP and performance consistency
df['avg_mAP'] = df[[f'mAP_{i}' for i in range(5, 10)]].mean(axis=1)
df['mAP_std'] = df[[f'mAP_{i}' for i in range(5, 10)]].std(axis=1)

# Calculate precision-recall balance
df['avg_precision'] = df[[f'Precision_{i}' for i in range(5, 10)]].mean(axis=1)
df['avg_recall'] = df[[f'Recall_{i}' for i in range(5, 10)]].mean(axis=1)
df['precision_recall_diff'] = df['avg_precision'] - df['avg_recall']

# Cluster images based on performance patterns
def cluster_performance(df):
    # Features for clustering
    features = df[[f'mAP_{i}' for i in range(5, 10)] + 
                  [f'Precision_{i}' for i in range(5, 10)] + 
                  [f'Recall_{i}' for i in range(5, 10)]].values
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(features)
    df['cluster'] = clusters
    
    # Analyze clusters
    cluster_stats = df.groupby('cluster').agg({
        'avg_mAP': 'mean',
        'mAP_std': 'mean',
        'avg_precision': 'mean',
        'avg_recall': 'mean',
        'precision_recall_diff': 'mean',
        'image_name': 'count'
    })
    
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    return df

# Perform clustering
df = cluster_performance(df)

# Identify different problematic categories separately
low_perf = df[df['avg_mAP'] < 0.3]  # Low performance
high_drop = df[(df['mAP_drop_5_to_6'] > 0.3) | 
               (df['mAP_drop_6_to_7'] > 0.3) | 
               (df['mAP_drop_7_to_8'] > 0.3) | 
               (df['mAP_drop_8_to_9'] > 0.3)]  # Localization issues
prec_rec_issues = df[(df['precision_recall_diff'].abs() > 0.3)]  # Precision-recall imbalance
high_var = df[df['mAP_std'] > 0.25]  # High variance (inconsistent performance)

# Save separate CSV files
low_perf.to_csv('low_performance_images2.csv', index=False)
high_drop.to_csv('high_localization_drop_images2.csv', index=False)
prec_rec_issues.to_csv('precision_recall_imbalance_images2.csv', index=False)
high_var.to_csv('high_variance_images2.csv', index=False)
false_positives = prec_rec_issues[prec_rec_issues['avg_precision'] < prec_rec_issues['avg_recall']]
false_positives.to_csv('false_positive_images.csv', index=False)
false_negatives = prec_rec_issues[prec_rec_issues['avg_precision'] > prec_rec_issues['avg_recall']]
false_negatives.to_csv('false_negative_images.csv', index=False)

# Function to visualize and save separate plots
def visualize_issues(df, problematic, filename, title, color):
    plt.figure(figsize=(15, 8))
    
    # Plot all images
    plt.scatter(df['avg_precision'], df['avg_recall'], 
                c=df['cluster'], cmap='viridis', alpha=0.6, label='All Images')
    
    # Highlight problematic images
    plt.scatter(problematic['avg_precision'], problematic['avg_recall'], 
                color=color, s=100, alpha=0.8, label='Problematic Images')
    
    plt.title(title)
    plt.xlabel('Average Precision')
    plt.ylabel('Average Recall')
    plt.legend()
    plt.grid()
    
    # Save instead of show
    plt.savefig(filename)
    plt.close()

# Generate and save separate plots for each issue
visualize_issues(df, low_perf, 'low_performance_plot2.png', 'Low Performance Images', 'red')
visualize_issues(df, high_drop, 'localization_issues_plot2.png', 'Localization Issues (High mAP Drop)', 'blue')
visualize_issues(df, prec_rec_issues, 'precision_recall_imbalance_plot2.png', 'Precision-Recall Imbalance', 'purple')
visualize_issues(df, high_var, 'high_variance_plot2.png', 'High Variance Images', 'orange')
