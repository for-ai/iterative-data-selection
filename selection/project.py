from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import umap
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import loguru

# Load the dataset
dataset_name = 'simonycl/p3_0.5_dataset'
dataset = load_dataset(dataset_name)
dataset_train = dataset['train']
loguru.logger.info("Dataset loaded.")

# Convert 'input' column to numpy array for efficient access
inputs_array = np.array(dataset_train['input'])

# Select a random sample of 1000 sentences from the dataset
random_indices = random.sample(range(len(inputs_array)), 10000)
sentences = inputs_array[random_indices].tolist()

# Initialize SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Time the encoding of sentences
start_time = time.time()
embeddings = model.encode(sentences, batch_size=256, device='cuda', show_progress_bar=True)
end_time = time.time()

# Log the time taken to encode sentences
loguru.logger.info(f"Sentence Transformer encoding took {end_time - start_time:.2f} seconds.")

# Initialize UMAP reducer
reducer = umap.UMAP(n_components=2)

# Time the UMAP reduction
start_time = time.time()
results = reducer.fit_transform(embeddings)
end_time = time.time()

# Log the time taken to fit_transform with UMAP
loguru.logger.info(f"UMAP fit_transform took {end_time - start_time:.2f} seconds.")

# Convert 'dataset' column to numpy array for efficient access and get colors for indices
datasets_array = np.array(dataset_train['dataset'])
colors = datasets_array[random_indices].tolist()

# Map color names to integers
unique_colors = set(colors)
color_categories = {name: idx for idx, name in enumerate(unique_colors)}
color_values = [color_categories[name] for name in colors]

# Create a scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(results[:, 0], results[:, 1], c=color_values, cmap='Spectral', alpha=0.7)

# Create a legend
legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Dataset")
plt.gca().add_artist(legend1)

# Map back the integers to original category names for the legend
plt.legend(handles=scatter.legend_elements()[0], labels=list(color_categories.keys()), loc='best', title='Dataset')

plt.title(f'UMAP projection of {dataset_name}')
plt.show()
plt.savefig('umap_projection.pdf', format='pdf', bbox_inches='tight')
