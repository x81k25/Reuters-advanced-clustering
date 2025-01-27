import pandas as pd
import numpy as np
from collections import Counter
import ast

################################################################################
# read in data and print basic info
################################################################################

# Read the parquet file
df = pd.read_parquet("data/reuters_with_clusters.parquet")

print("=" * 50)
print("BASIC DATAFRAME INFO")
print("=" * 50)
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nColumn names:")
for col in df.columns:
    print(f"- {col}")

print("\n" + "=" * 50)
print("DATATYPES")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("NUMERIC COLUMNS STATISTICS")
print("=" * 50)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_cols) > 0:
    print(df[numeric_cols].describe())
else:
    print("No numeric columns found")

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found")

print("\n" + "=" * 50)
print("SAMPLE DATA")
print("=" * 50)
print(df.head())

################################################################################
#
# generates metrics for comparison
#
# important metrics:
#   -Category Concentration:
#     -For a given category, it's the number of its occurrences in its most
#     common cluster divided by the total number of occurrences of that category
#     -For example, if a category appears 200 times total across all documents,
#     and its most frequent cluster contains 120 of those occurrences, the
#     concentration would be 120/200 = 60%
#     -Higher concentration means the category is more concentrated in a single
#     cluster rather than spread across many
#   -Cluster Purity:
#     -For a given cluster, it's the number of occurrences of the most common
#     category divided by the total number of category assignments in that
#     cluster
#     -For example, if a cluster has 100 documents, and some documents have
#     multiple categories, let's say total of 150 category assignments, and
#     the most frequent category appears 90 times, the purity would be
#     90/150 = 60%
#     -Higher purity means the cluster is more focused on a single category
#
###############################################################################

# Basic stats
n_clusters = len(df['cluster_label'].unique())
n_docs = len(df)
n_categories = len(
    set(cat for cats in df['categories'].values for cat in np.array(cats)))

print(f"Documents: {n_docs}")
print(f"Clusters: {n_clusters}")
print(f"Unique categories: {n_categories}")

# Cluster metrics
cluster_stats = {}
for cluster in df['cluster_label'].unique():
    cluster_docs = df[df['cluster_label'] == cluster]
    all_categories = [cat for cats in cluster_docs['categories'].values for cat
                      in np.array(cats)]
    category_counts = Counter(all_categories)

    cluster_stats[cluster] = {
        'size': len(cluster_docs),
        'size_percentage': len(cluster_docs) / n_docs * 100,
        'unique_categories': len(category_counts),
        'dominant_category': max(category_counts.items(), key=lambda x: x[1])[
            0] if category_counts else 'none',
        'purity': max(category_counts.values()) / len(
            all_categories) * 100 if all_categories else 0
    }

# Category metrics
category_stats = {}
all_categories = set(
    cat for cats in df['categories'].values for cat in np.array(cats))
for category in all_categories:
    category_docs = df[
        df['categories'].apply(lambda x: category in np.array(x))]
    cluster_counts = Counter(category_docs['cluster_label'])

    category_stats[category] = {
        'occurrences': len(category_docs),
        'clusters_spread': len(cluster_counts),
        'main_cluster': max(cluster_counts.items(), key=lambda x: x[1])[
            0] if cluster_counts else 'none',
        'concentration': max(cluster_counts.values()) / len(
            category_docs) * 100 if cluster_counts else 0
    }

# Print top clusters by size
print("\nTop 10 clusters by size:")
top_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['size'],
                      reverse=True)[:10]
for cluster, stats in top_clusters:
    print(f"\nCluster {cluster}:")
    print(f"Size: {stats['size']} ({stats['size_percentage']:.1f}%)")
    print(f"Dominant category: {stats['dominant_category']}")
    print(f"Purity: {stats['purity']:.1f}%")
    print(f"Unique categories: {stats['unique_categories']}")

# Print top categories by frequency
print("\nTop 10 categories by frequency:")
top_categories = sorted(category_stats.items(),
                        key=lambda x: x[1]['occurrences'], reverse=True)[:10]
for category, stats in top_categories:
    print(f"\n{category}:")
    print(f"Occurrences: {stats['occurrences']}")
    print(f"Spread across {stats['clusters_spread']} clusters")
    print(f"Main cluster: {stats['main_cluster']}")
    print(f"Concentration: {stats['concentration']:.1f}%")

# Overall quality metrics
avg_purity = np.mean([stats['purity'] for stats in cluster_stats.values()])
avg_concentration = np.mean(
    [stats['concentration'] for stats in category_stats.values()])
print(f"\nAverage cluster purity: {avg_purity:.1f}%")
print(f"Average category concentration: {avg_concentration:.1f}%")

total_docs = len(df)
cluster_neg1_size = len(df[df['cluster_label'] == -1])
print(f"Percentage of documents in cluster -1: {(cluster_neg1_size/total_docs)*100:.1f}%")

# Let's also look at how many positive clusters we have and their average size
positive_clusters = df[df['cluster_label'] != -1]['cluster_label'].unique()
print(f"Number of positive clusters: {len(positive_clusters)}")
print(f"Average size of positive clusters: {len(df[df['cluster_label'] != -1])/len(positive_clusters):.1f}")

print("\nTop clusters by size (excluding -1):")
top_positive_clusters = sorted(
    [(c, stats) for c, stats in cluster_stats.items() if c != -1],
    key=lambda x: x[1]['size'],
    reverse=True
)[:10]

for cluster, stats in top_positive_clusters:
    print(f"\nCluster {cluster}:")
    print(f"Size: {stats['size']} ({stats['size_percentage']:.1f}%)")
    print(f"Dominant category: {stats['dominant_category']}")
    print(f"Purity: {stats['purity']:.1f}%")

################################################################################
# end of analysis_of_results.py
################################################################################
