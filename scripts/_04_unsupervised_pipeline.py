# Import required libraries
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from umap import UMAP
from hdbscan import HDBSCAN
import re
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, message="'force_all_finite' was renamed to 'ensure_all_finite'")

################################################################################
# test GPU availability
################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

################################################################################
# load model parameters defined through gridsearch
################################################################################

model_params = {
   'umap': {
       'n_components': 20,  # Number of dimensions to reduce to
       'n_neighbors': 10,   # Size of local neighborhood
       'min_dist': 0.2,     # Minimum distance between points in low dimensional space
       'metric': 'cosine'   # Distance metric for comparing points
   },
   'hdbscan': {
       'min_cluster_size': 10,  # Minimum size for a cluster
       'min_samples': 5,        # Number of samples in neighborhood for core points
       'cluster_selection_epsilon': 0.1,  # Distance threshold for cluster membership
       'metric': 'euclidean',   # Distance metric for comparing points
       'cluster_selection_method': 'eom'  # Method for selecting clusters
   }
}

################################################################################
# load and preprocess the data
################################################################################

def load_and_preprocess_data(file_path):
    """Load data from parquet file and clean the text"""
    df = pd.read_parquet(file_path)
    def clean_text(text):
        # Remove extra whitespace and standardize text
        text = re.sub(r'\s+', ' ', str(text).strip())
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.lower()
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

df = load_and_preprocess_data('data/reuters_data.parquet')

################################################################################
# get embeddings via BERT
################################################################################

def get_bert_embeddings(texts, batch_size=32):
    """Convert text to BERT embeddings in batches"""
    texts = [str(text) if text is not None else "" for text in texts]
    print("Initializing BERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    model = model.to(device)
    model.eval()
    all_embeddings = []

    print("Generating BERT embeddings...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        # Tokenize text and move to GPU if available
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generate embeddings without computing gradients
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Generate BERT embeddings for the text
embeddings = get_bert_embeddings(df['cleaned_text'].values)
print(f"Embeddings shape: {embeddings.shape}")

################################################################################
# dimensionality reduction via UMAP
################################################################################

def reduce_dimensions(embeddings):
    """Reduce dimensionality of embeddings using UMAP"""
    print("Performing UMAP dimensionality reduction...")
    umap_reducer = UMAP(
        n_neighbors=model_params['umap']['n_neighbors'],
        min_dist=model_params['umap']['min_dist'],
        n_components=model_params['umap']['n_components'],
        random_state=None,
        metric=model_params['umap']['metric'],
        n_jobs=-1
    )
    return umap_reducer.fit_transform(embeddings)

reduced_embeddings = reduce_dimensions(embeddings)
print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

################################################################################
# clustering via HDBSCAN
################################################################################

def perform_clustering(reduced_embeddings):
    """Cluster the reduced embeddings using HDBSCAN"""
    print("Performing HDBSCAN clustering...")
    clusterer = HDBSCAN(
        min_cluster_size=model_params['hdbscan']['min_cluster_size'],
        min_samples=model_params['hdbscan']['min_samples'],
        metric=model_params['hdbscan']['metric'],
        cluster_selection_method=model_params['hdbscan']['cluster_selection_method'],
        cluster_selection_epsilon=model_params['hdbscan']['cluster_selection_epsilon'],
        prediction_data=True,
        core_dist_n_jobs=-1,  # Added this for parallelism
        allow_single_cluster=True  # This helps avoid the force_all_finite warning
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    # Calculate and display clustering statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Found {n_clusters} clusters")
    print(f"Number of noise points: {n_noise} ({n_noise / len(cluster_labels) * 100:.2f}%)")
    return cluster_labels

cluster_labels = perform_clustering(reduced_embeddings)
print(f"Clustering complete. Shape of labels: {cluster_labels.shape}")

################################################################################
# create final DataFrame with UMAP components and cluster labels
################################################################################

# Create DataFrame with UMAP components and cluster labels
umap_df = pd.DataFrame(
    reduced_embeddings,
    columns=[f'umap_{i+1}' for i in range(model_params['umap']['n_components'])]
)
umap_df['cluster_label'] = cluster_labels
final_df = pd.concat([df, umap_df], axis=1)

################################################################################
# format and save results
################################################################################

# format the final_df for saving
export_df = final_df

# remove id and cleaned_text columns from export_df
export_df = export_df.drop(columns=['id', 'cleaned_text'])

# rename categories to category_lables
export_df = export_df.rename(columns={'categories': 'category_labels'})

# add new column category_label_1 that shows the first category in the list
export_df['category_label_1'] = export_df['category_labels'].apply(lambda x: x[0] if isinstance(x, np.ndarray) and x.size > 0 else None)

# change the order of the columns to categories, cluster_label, text, and then the umap dimensions
cols = ['category_labels', 'category_label_1', 'cluster_label', 'text'] + [col for col in final_df.columns if col.startswith('umap_')]
export_df = export_df[cols]

# save
export_df.to_parquet('data/reuters_with_clusters.parquet')

# select 10 random rows from final_df
df_sample = export_df.sample(10)
df_sample.to_csv('data/clustered_data_sample.csv', index=False)

################################################################################
# end of unsupervised_pipeline.py
################################################################################
