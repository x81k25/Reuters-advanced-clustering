# Reuters advanced clustering

This project implements an unsupervised clustering pipeline using UMAP for dimensionality reduction and HDBSCAN for clustering. The pipeline processes text data, generates BERT embeddings, reduces dimensionality, and performs clustering to identify meaningful groups in the data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Methodology](#methodology)
  - [Dataset](#dataset)
  - [Technical Pipeline](#technical-pipeline)
    - [Text Preprocessing](#1-text-preprocessing)
    - [Text Embeddings](#2-text-embeddings)
    - [Dimensionality Reduction](#3-dimensionality-reduction)
    - [Clustering](#4-clustering)
  - [Model Selection and Hyperparameter Tuning](#model-selection-and-hyperparameter-tuning)
  - [Analysis and Visualization](#analysis-and-visualization)
  
## Installation

To install the required dependencies, establish venv, and run:

```bash
pip install -r requirements.txt
```

*Note: The current requirement.txt configuration is personalized to my hardware and may need adjustment for your system for GPU acceleration. However, all code should run an any system as currently configured, however GPU acceleration is not supported unviersally.

## Project Structure

- `scripts/`: Contains the main scripts for running the clustering pipeline and parameter optimization.
- `data/`: Directory containing input data and generated results.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and visualization.
- `requirements.txt`: List of required Python packages.
- `.gitignore`: Git ignore file to exclude unnecessary files from version control.
- `README.md`: Project documentation file.

## Pipeline Overview

1. **Data Loading and Preprocessing**:
   - Load data from a parquet file.
   - Clean the text data by removing extra whitespace and standardizing the text.

2. **BERT Embeddings**:
   - Generate BERT embeddings for the cleaned text data using the `transformers` library.

3. **Dimensionality Reduction**:
   - Reduce the dimensionality of the embeddings using UMAP.

4. **Clustering**:
   - Perform clustering on the reduced embeddings using HDBSCAN.

5. **Results Formatting and Saving**:
   - Create a DataFrame with UMAP components and cluster labels.
   - Save the results to a parquet file and export a sample to a CSV file.

# Methodology

## Dataset
This project utilizes the Reuters-21578 dataset, a widely-used corpus in text classification and natural language processing research. The dataset consists of 21,578 news articles from the Reuters newswire (1987), with the following characteristics:

- Documents are manually labeled with multiple categories (multi-label classification)
- Categories include topics such as earnings, acquisitions, grain, crude oil, etc.
- Labels were assigned by Reuters Ltd. personnel and refined by Carnegie Group Inc.
- Dataset exhibits natural class imbalance
- Includes both training and test splits

## Technical Pipeline

### 1. Text Preprocessing
- Cleaned text by removing extra whitespace
- Standardized text to lowercase
- Removed special characters while preserving basic punctuation
- Applied regex-based cleaning using pattern `[^\w\s.,!?-]`

### 2. Text Embeddings
Used BERT (Bidirectional Encoder Representations from Transformers) for document embeddings:

- Model: `bert-base-uncased` (12 layers, 768 hidden dimensions)
- Generates 768-dimensional contextual embeddings
- Used [CLS] token embedding as document representation
- Implemented batch processing (batch_size=32) for efficiency
- Applied padding and truncation to 512 tokens

Key advantages of BERT embeddings:
- Contextual understanding of words
- Pre-trained knowledge transfer
- Captures rich semantic relationships
- Handles polysemy effectively

### 3. Dimensionality Reduction
Implemented UMAP (Uniform Manifold Approximation and Projection) with the following optimized parameters:

- `n_components`: 20 (output dimensions)
- `n_neighbors`: 10 (local neighborhood size)
- `min_dist`: 0.2 (minimum distance between points)
- `metric`: cosine (distance metric)

UMAP was chosen for:
- Superior preservation of both local and global structure
- Computational efficiency with large datasets
- Strong theoretical foundation in manifold learning
- Effective density preservation

### 4. Clustering
Applied HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) with parameters:

- `min_cluster_size`: 10
- `min_samples`: 5
- `cluster_selection_epsilon`: 0.1
- `metric`: euclidean
- `cluster_selection_method`: eom (Excess of Mass)

Benefits of HDBSCAN:
- No need to specify number of clusters a priori
- Natural handling of noise points
- Adapts to varying cluster densities
- Hierarchical structure preservation

## Model Selection and Hyperparameter Tuning

Hyperparameters were optimized through grid search, evaluating combinations of:

UMAP parameters:
```python
{
    'n_components': [10, 20, 30],
    'n_neighbors': [10, 15, 20],
    'min_dist': [0.1, 0.2],
    'metric': ['cosine']
}
```

HDBSCAN parameters:
```python
{
    'min_cluster_size': [5, 10, 15],
    'min_samples': [3, 5, 7],
    'cluster_selection_epsilon': [0.1, 0.2, 0.3],
    'metric': ['euclidean'],
    'cluster_selection_method': ['eom']
}
```

Final parameters were selected based on:
- Maximizing cluster purity relative to input categories
- Minimizing overall noise points
- Balancing computational efficiency with performance

## Analysis and Visualization

Results were analyzed using multiple visualization techniques:

1. 3D UMAP visualization using Plotly
   - Interactive scatter plot of first three UMAP components
   - Color-coded by cluster labels
   - Hover data includes category labels

2. Cluster quality metrics:
   - Cluster size distribution
   - Cluster purity measurements
   - Majority category analysis per cluster

3. Statistical visualizations:
   - Distribution of cluster purities
   - Cluster size vs purity relationships
   - Analysis of the whole entirety cluster space
   - Drill-downed visualization of select clusters

