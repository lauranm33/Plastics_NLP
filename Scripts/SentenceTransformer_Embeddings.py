# =============================================================================
# Package Import
# =============================================================================
import concurrent.futures, numpy as np, os, pandas as pd, re, sys
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# NLP_utils.py :
SCRIPTS_DIR = r"D:\MS_Thesis\Scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
# Selective import from NLP_utils.py, imports in order of use
from NLP_utils import (process_chunk, fix_embedding_format, 
                       concatenate_embedding_chunks, compute_top_k_similarities,
                       save_sbert_performance, plot_similarity_distribution,
                       plot_gaussian_similarity_distribution)
#%%============================================================================
# Process Chunks
# =============================================================================
SBERT_DIR = r"D:/MS_Thesis/SentenceTransformer"
input_csv = r"D:/MS_Thesis/API_Queries/Abstracts/All_Abstracts.csv"
output_folder = f"{SBERT_DIR}/Embedding_Chunks"
model = SentenceTransformer("all-MiniLM-L6-v2")
chunksize = 1000

# =============================================================================
# Chunk execution (process_chunk execution)
# =============================================================================
for i, chunk in enumerate(pd.read_csv(input_csv, chunksize = chunksize)):
    process_chunk(model, output_folder, chunk, i)
    
# =============================================================================
# Combine all processed chunks (concatenate_embedding_chunks execution)
# =============================================================================
df_all = concatenate_embedding_chunks(output_folder, SBERT_DIR)
df_all["Embedding"] = df_all["Embedding"].apply(fix_embedding_format)
# Convert to proper NumPy array
embeddings = np.stack(df_all["Embedding"].values)

# =============================================================================
# compute_top_k_similarities among embeddings execution
# =============================================================================
top_k_indices_chunked, top_k_values_chunked = compute_top_k_similarities(embeddings)

# =============================================================================
# Save to CSV (save_sbert_performance execution)
# =============================================================================
sbert_performance_path = f"{SBERT_DIR}/SentenceTransformer_Performance.csv"
save_sbert_performance(top_k_indices_chunked, 
                       top_k_values_chunked, 
                       sbert_performance_path)

# =============================================================================
# Evaluate the distribution of sentence embeddings (should be symmetric)
# =============================================================================
# Desirable value should be from 0.5 < x < 1.0
avg_sim = np.mean(top_k_values_chunked)
print(f"Average Top-K Similarity: {avg_sim:.4f}") 
# Standard deviation of similarity scores (should be small)
sim_std = np.std(top_k_values_chunked)
print(f"Standard Deviation of Top-K Similarity: {sim_std:.4f}")

#%%============================================================================
# Saving similarity scores to entire dataframe
# =============================================================================
dataset = pd.read_csv(f"{SBERT_DIR}/All_Embeddings.csv")
df = pd.read_csv(f"{SBERT_DIR}/SentenceTransformer_Performance.csv")
df_f = pd.concat([dataset, df], axis = 1)
df_f.drop_duplicates(subset = ["Index"], keep = "first")
df_f.drop(columns=["Unnamed: 0"], errors = "ignore", inplace = True)
df_f.to_csv(f"{SBERT_DIR}/SentenceEmbedding_Similarities.csv")

#%%============================================================================
# Plot similarity distribution - should have a symmetrical distribution
# =============================================================================
# Save figure paths
fig_path = r"D:/MS_Thesis/Figures"
# Unsmoothed distribution path
sim_path = f"{fig_path}/TopK_Similarity_Distribution.png" 
# Using Gaussian smoothing technique path
gauss_path = f"{fig_path}/TopK_Similarity_Distribution_Gaussian_Smoothed.png"

# Execute plot_similarity_distribution & plot_gaussian_similarity_distribution
plot_similarity_distribution(top_k_values_chunked, sim_path)
plot_gaussian_similarity_distribution(top_k_values_chunked, gauss_path)