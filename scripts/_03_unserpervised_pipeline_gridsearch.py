import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import itertools
import warnings
import os

# Try to import GPU-accelerated UMAP
try:
	print("Attempting to import CUML UMAP...")
	from cuml.manifold import UMAP
	from cuml.common.handle import Handle

	# Test if CUDA is actually available
	handle = Handle()
	print("CUML successfully imported and CUDA handle created")
	GPU_AVAILABLE = True
except ImportError as e:
	print(f"Failed to import CUML: {str(e)}")
	from umap import UMAP

	GPU_AVAILABLE = False
except Exception as e:
	print(f"Other error occurred: {str(e)}")
	from umap import UMAP

	GPU_AVAILABLE = False

print(f"GPU acceleration available: {GPU_AVAILABLE}")

warnings.filterwarnings('ignore')

# Try to import GPU-accelerated UMAP
try:
	from cuml.manifold import UMAP
	from cuml.common.handle import Handle

	GPU_AVAILABLE = True
except ImportError:
	from umap import UMAP

	GPU_AVAILABLE = False

from hdbscan import HDBSCAN
from collections import Counter


def load_data(file_path):
	"""Load and prepare data from parquet file."""
	df = pd.read_parquet(file_path)

	# Convert text to string and handle any NULL values
	texts = [str(text) if pd.notna(text) else "" for text in df['text'].values]

	# Ensure categories are in list format
	categories = df['categories'].values

	return texts, categories


def generate_bert_embeddings(texts, batch_size=32,
							 model_name='bert-base-uncased'):
	"""Generate BERT embeddings using GPU acceleration."""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModel.from_pretrained(model_name)
	model = model.to(device)
	model.eval()

	all_embeddings = []

	for i in tqdm(range(0, len(texts), batch_size),
				  desc="Generating BERT embeddings"):
		batch_texts = texts[i:i + batch_size]

		inputs = tokenizer(batch_texts, padding=True, truncation=True,
						   max_length=512, return_tensors="pt")
		inputs = {k: v.to(device) for k, v in inputs.items()}

		with torch.no_grad():
			outputs = model(**inputs)
			embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
			all_embeddings.append(embeddings)

	return np.vstack(all_embeddings)


def calculate_cluster_purity(cluster_labels, true_labels, noise_penalty=0.5):
	"""Calculate cluster purity with noise penalty."""
	n_samples = len(cluster_labels)
	n_noise = sum(1 for label in cluster_labels if label == -1)
	noise_percentage = n_noise / n_samples

	unique_clusters = set(cluster_labels)
	if -1 in unique_clusters:
		unique_clusters.remove(-1)

	if not unique_clusters:
		return 0.0, {}, {'noise_percentage': 1.0, 'n_clusters': 0}

	cluster_stats = {}
	total_purity = 0

	for cluster in unique_clusters:
		cluster_indices = np.where(cluster_labels == cluster)[0]
		cluster_label_lists = [true_labels[i] for i in cluster_indices]
		all_cluster_labels = [label for label_list in cluster_label_lists for
							  label in label_list]
		label_counts = Counter(all_cluster_labels)
		dominant_label = max(label_counts.items(), key=lambda x: x[1])

		pure_docs = sum(1 for label_list in cluster_label_lists if
						dominant_label[0] in label_list)
		cluster_purity = pure_docs / len(cluster_indices)

		cluster_stats[cluster] = {
			'purity': cluster_purity,
			'dominant_label': dominant_label[0],
			'size': len(cluster_indices)
		}

		total_purity += cluster_purity * len(cluster_indices)

	avg_purity = total_purity / (
			n_samples - n_noise) if n_samples > n_noise else 0
	final_purity = avg_purity * (1 - noise_penalty * noise_percentage)

	metrics = {
		'noise_percentage': noise_percentage,
		'n_clusters': len(unique_clusters),
		'avg_cluster_size': (n_samples - n_noise) / len(unique_clusters),
		'largest_cluster_size': max(
			stats['size'] for stats in cluster_stats.values()),
		'smallest_cluster_size': min(
			stats['size'] for stats in cluster_stats.values())
	}

	return final_purity, cluster_stats, metrics


def run_pipeline(embeddings, true_labels, umap_params, hdbscan_params):
	"""Run clustering pipeline with current parameters."""
	if GPU_AVAILABLE:
		with Handle():
			umap_reducer = UMAP(**umap_params)
			reduced_embeddings = umap_reducer.fit_transform(embeddings)
	else:
		umap_reducer = UMAP(**umap_params)
		reduced_embeddings = umap_reducer.fit_transform(embeddings)

	clusterer = HDBSCAN(**hdbscan_params)
	cluster_labels = clusterer.fit_predict(reduced_embeddings)

	purity_score, cluster_stats, metrics = calculate_cluster_purity(
		cluster_labels, true_labels
	)

	return purity_score, cluster_stats, metrics, cluster_labels


def optimize_clustering(
	data_path,
	param_grid,
	embeddings_cache='data/bert_embeddings.npy'
):
	"""
	Complete pipeline from data loading through parameter optimization.

	Args:
		data_path: Path to parquet file
		param_grid: Dictionary of parameters to try
		embeddings_cache: Path to save/load BERT embeddings

	Returns:
		DataFrame with results for each parameter combination
	"""
	print("Loading data...")
	texts, labels = load_data(data_path)

	if os.path.exists(embeddings_cache):
		print("Loading cached BERT embeddings...")
		embeddings = np.load(embeddings_cache)
	else:
		print("Generating BERT embeddings...")
		embeddings = generate_bert_embeddings(texts)
		np.save(embeddings_cache, embeddings)

	umap_keys, umap_values = zip(*param_grid['umap'].items())
	hdbscan_keys, hdbscan_values = zip(*param_grid['hdbscan'].items())

	umap_combinations = list(itertools.product(*umap_values))
	hdbscan_combinations = list(itertools.product(*hdbscan_values))

	results = []
	total_combinations = len(umap_combinations) * len(hdbscan_combinations)

	print(f"Testing {total_combinations} parameter combinations...")
	with tqdm(total=total_combinations) as pbar:
		for umap_combo in umap_combinations:
			for hdbscan_combo in hdbscan_combinations:
				umap_params = dict(zip(umap_keys, umap_combo))
				hdbscan_params = dict(zip(hdbscan_keys, hdbscan_combo))

				purity_score, cluster_stats, metrics, _ = run_pipeline(
					embeddings, labels, umap_params, hdbscan_params
				)

				result = {
					**{f'umap_{k}': v for k, v in umap_params.items()},
					**{f'hdbscan_{k}': v for k, v in hdbscan_params.items()},
					'purity_score': purity_score,
					**metrics
				}

				results.append(result)
				pbar.update(1)

	results_df = pd.DataFrame(results)
	results_df = results_df.sort_values('purity_score', ascending=False)

	return results_df


# run test of main function to ensure it works
# Minimal parameter grid for testing
test_param_grid = {
    'umap': {
        'n_components': [10],
        'n_neighbors': [15],
        'min_dist': [0.1],
        'metric': ['cosine']
    },
    'hdbscan': {
        'min_cluster_size': [5],
        'min_samples': [3],
        'cluster_selection_epsilon': [0.1],
        'metric': ['euclidean'],
        'cluster_selection_method': ['eom']
    }
}

# Run test with minimal parameters
test_results = optimize_clustering(
    data_path='data/reuters_data_test.parquet',
    param_grid=test_param_grid,
    embeddings_cache='data/test_bert_embeddings.npy'
)

print("\nTest results:")
print(test_results)


# Define parameter grid
param_grid = {
	'umap': {
		'n_components': [10, 20, 30],
		'n_neighbors': [10, 15, 20],
		'min_dist': [0.1, 0.2],
		'metric': ['cosine']
	},
	'hdbscan': {
		'min_cluster_size': [5, 10, 15],
		'min_samples': [3, 5, 7],
		'cluster_selection_epsilon': [0.1, 0.2, 0.3],
		'metric': ['euclidean'],
		'cluster_selection_method': ['eom']
	}
}

# Run optimization
results_df = optimize_clustering(
	data_path='data/reuters_data.parquet',
	param_grid=param_grid
)

# Save results
results_df.to_csv('data/clustering_results.csv', index=False)

# read in results from csv
results_df = pd.read_csv('data/clustering_results.csv')

# Print top 5 parameter combinations
print("\nTop 5 parameter combinations:")
print(results_df.head())