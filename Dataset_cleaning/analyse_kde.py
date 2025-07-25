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

# Function to plot KDE for mAP values and save as PNG
def plot_maps_kde(df):
    plt.figure(figsize=(15, 10))
    
    # Plot KDE for each mAP
    for i in range(5, 10):
        data = df[f'mAP_{i}'].values
        data = data[~np.isnan(data)]
        
        # Create KDE
        kde = gaussian_kde(data)
        x_vals = np.linspace(0, 1, 1000)
        y_vals = kde(x_vals)
        
        # Find valleys (local minima)
        valleys = []
        for j in range(1, len(y_vals)-1):
            if y_vals[j-1] > y_vals[j] < y_vals[j+1]:
                valleys.append(x_vals[j])
        
        # Plot
        plt.plot(x_vals, y_vals, label=f'mAP @ 0.{i}')
        for valley in valleys:
            plt.axvline(x=valley, color='r', linestyle='--', alpha=0.3)
        
        print(f"mAP @ 0.{i} - Valleys at: {valleys}")
    
    plt.title('KDE of mAP Values Across IoU Thresholds')
    plt.xlabel('mAP Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    
    # Save instead of show
    plt.savefig('kde_plot.png')
    plt.close()

# Generate KDE plot and save
plot_maps_kde(df)

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

# Identify problematic images based on thresholds
def identify_problematic(df):
    # Images with consistently low performance
    low_perf = df[df['avg_mAP'] < 0.3]
    
    # Images with high mAP drop (localization issues)
    high_drop = df[(df['mAP_drop_5_to_6'] > 0.3) | 
                   (df['mAP_drop_6_to_7'] > 0.3) | 
                   (df['mAP_drop_7_to_8'] > 0.3) | 
                   (df['mAP_drop_8_to_9'] > 0.3)]
    
    # Images with precision-recall imbalance
    prec_rec_issues = df[(df['precision_recall_diff'].abs() > 0.3)]
    
    # Images with high variance (inconsistent performance)
    high_var = df[df['mAP_std'] > 0.25]
    
    # Combine all problematic images
    problematic = pd.concat([low_perf, high_drop, prec_rec_issues, high_var]).drop_duplicates()
    
    print(f"\nIdentified {len(problematic)} potentially problematic images")
    
    return problematic

# Get problematic images
problematic_images = identify_problematic(df)

# Save results
problematic_images.to_csv('problematic_images.csv', index=False)
df.to_csv('analyzed_mAP_results.csv', index=False)

# Visualization of problematic images (saved as PNG)
def visualize_problems(problematic, df):
    plt.figure(figsize=(15, 8))
    
    # Plot all images
    plt.scatter(df['avg_precision'], df['avg_recall'], 
               c=df['cluster'], cmap='viridis', alpha=0.6, label='All Images')
    
    # Highlight problematic images
    plt.scatter(problematic['avg_precision'], problematic['avg_recall'], 
               color='red', s=100, alpha=0.8, label='Problematic Images')
    
    plt.title('Precision vs Recall with Problematic Images Highlighted')
    plt.xlabel('Average Precision')
    plt.ylabel('Average Recall')
    plt.legend()
    plt.grid()
    
    # Save instead of show
    plt.savefig('precision_recall_plot2.png')
    plt.close()

# Generate and save Precision-Recall plot
visualize_problems(problematic_images, df)
