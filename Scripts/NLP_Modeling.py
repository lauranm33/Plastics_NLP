#%%============================================================================
# Package Imports:
# =============================================================================
# Basic Functionality Packages
import joblib, os, numpy as np, pandas as pd, sys, re
from typing import List, Union    
from warnings import warn

# ML Packages
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
import hdbscan, optuna, umap
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer   
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                             calinski_harabasz_score)
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# Plotting packages:
import matplotlib.pyplot as plt, matplotlib.collections as mc, seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'
from plotly.offline import plot
import plotly.graph_objects as go
import textwrap, webbrowser

# Color palettes used for thesis figures from NLP_utils.py
SCRIPTS_DIR = r"D:\MS_Thesis\Scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from NLP_utils import get_palette
gem_palette = get_palette("gem")
datamapplot_palette = get_palette("datamapplot")
temporal_colors = get_palette("temporal")
temporal_palette = sns.color_palette(temporal_colors)

# NLP_utils.py
from NLP_utils import (fix_embedding_format, preprocess_document, 
                       preprocess_corpus, objective, 
                       plot_cluster_keyword_trends)
#%%============================================================================
# Configuration:
# =============================================================================
# Filepaths
MAIN_DIR = r"D:/MS_Thesis"
SCRIPTS_DIR = f"{MAIN_DIR}/Scripts"
SBERT_DIR = f"{MAIN_DIR}/SentenceTransformer"
VISUALS_DIR = f"{MAIN_DIR}/Figures"
CLUSTERS_DIR = f"{MAIN_DIR}/Clusters"
CLUSTER_CSV_DIR = f"{CLUSTERS_DIR}/Cluster_CSVs"
MODELS_DIR = f"{MAIN_DIR}/Models"

# Optuna Configuration
CONFIG = {
    "save_file" : f"{SCRIPTS_DIR}/NLP_Study_Final.pkl", 
    "dataset_path" : f"{SBERT_DIR}/All_Embeddings.csv",
    "checkpoints" : {"data_loaded": False,
                    "optuna_done": False,
                    "umap_done": False,
                    "hdbscan_done": False,
                    "BERTopic_done": False}}
#%%============================================================================
# Load in data if it exists:
#==============================================================================
# Data load-in checkpoint
data        = joblib.load(CONFIG["save_file"])
df          = data.get("df")
embeddings  = data.get("embeddings")
docs        = data.get("docs")
CONFIG["checkpoints"]["data_loaded"] = True
# Optuna checkpoint
best_params = data.get("best_params")
best_trial  = data.get("best_trial")
sil_score   = data.get("sil_score")
db_score    = data.get("db_score")
ch_score    = data.get("ch_score")
CONFIG["checkpoints"]["optuna_done"] = True
# UMAP fit checkpoiint
best_umap       = data.get("best_umap")
umap_embeddings = data.get("umap_embeddings")
CONFIG["checkpoints"]["umap_done"] = True
# HDBSCAN fit checkpoint
best_hdbscan = data.get("best_hdbscan")
labels       = data.get("labels")
CONFIG["checkpoints"]["hdbscan_done"] = True
# BERTopic fit checkpoint
topic_model = data.get("topic_model")
CONFIG["checkpoints"]["BERTopic_done"] = True
#%%============================================================================
# Preprocess dataset using NLP_utils
# =============================================================================
# Read in dataset w/ SBERT embeddings
df = pd.read_csv(CONFIG["dataset_path"])  
# Drop extra index columns if they exist
df.drop(columns = ["Unnamed: 0"], errors = "ignore", inplace=True)
# Convert embeddings from string to np.array
df["Embedding"] = df["Embedding"].apply(fix_embedding_format)
# Embeddings as a NumPy array
embeddings = np.stack(df["Embedding"].values) 
# Extract raw abstracts (not processed)
abstracts = df["Abstract"].tolist() 
# Lowercase, remove special characters, tokenize, lemmatize, & filter
docs, stopwords_set = preprocess_corpus(
    abstracts,
    f"{MAIN_DIR}/Stopwords/combined_stopwords.txt")

# Save checkpoint: Data load in
try:              
    CONFIG['checkpoints']['data_loaded'] = True
    joblib.dump({"df" : df, 
                 "embeddings" : embeddings,
                 "docs" : docs, 
                 "config" : CONFIG}, 
                CONFIG["save_file"])
    print("Data successfully loaded and preprocessed!")

except Exception as e:
    print(f"Failed to save data: {str(e)}")
#%%============================================================================
# Executive Optuna Optimization & Save Checkpoint (NLP_utils.py)
# =============================================================================
# Define study
study = optuna.create_study(directions = ["maximize", "minimize", "maximize"],
                            sampler = optuna.samplers.NSGAIISampler())
# Execute study
study.optimize(lambda trial: objective(trial, embeddings), 
               n_trials = 100, show_progress_bar = True)

# Results of best trial
best_trial  = study.best_trials[0]
sil_score   = best_trial.values[0]
db_score    = best_trial.values[1]
ch_score    = best_trial.values[2]
best_params = best_trial.params
print("Best trial:")
print(f"  Silhouette Score: {best_trial.values[0]}")
print(f"  Davies-Bouldin Index: {-best_trial.values[1]}")
print(f"  Calinski-Harabasz Index: {best_trial.values[2]}")
print(f"  Params: {best_trial.params}")

# Save checkpoint: Optuna optimization
try:              
    CONFIG["checkpoints"]["optuna_done"] = True
    joblib.dump({
        "df" : df,
        "embeddings" : embeddings,
        "docs" : docs,
        "best_trial" : best_trial,
        "sil_score" : sil_score,
        "db_score" : db_score,
        "ch_score" : ch_score,
        "config" : CONFIG}, 
        CONFIG["save_file"])
    print("Optuna optimization complete!")

except Exception as e:
    print(f"Optuna optimization failed: {str(e)}")
#%%============================================================================
# Pareto Plot for Optuna
# =============================================================================
# Convert Optuna trials into pd.df with selected attributes
df_optuna = study.trials_dataframe(
    attrs = ("number", "value", "params", "state"))

# Extract each objective into its own column (for clarity)
df_optuna["silhouette"] = df_optuna["values_0"]
df_optuna["dbi"]        = df_optuna["values_1"]
df_optuna["ch_score"]   = df_optuna["values_2"]

# Rename columns to shorter labels & extract hyperparameter names
df_optuna = df_optuna.rename(columns = {
    "silhouette" : "SS",
    "dbi" : "DBI", 
    "ch_score" : "CHI",
    "params_umap_n_neighbors" : "n_neighbors",
    "params_umap_min_dist" : "min_dist",
    "min_dist" : "min_dist",
    "params_hdbscan_min_cluster_size" : "min_cluster_size",
    "params_hdbscan_min_samples" : "min_samples",
    "params_hdbscan_cluster_selection_epsilon": "epsilon"})

# Identify the row corresponding to the best trial based on saved scores
df_optuna["Best"] = (
    (df_optuna["SS"].round(6) == round(sil_score, 6)) &
    (df_optuna["DBI"].round(6) == round(db_score, 6)) &
    (df_optuna["CHI"].round(6) == round(ch_score, 6)))

# Initialize 3D scatter figure
fig = go.Figure()

# Plot all non-best trials as semi-transparent circle markers
df_all = df_optuna[~df_optuna["Best"]]

fig.add_trace(go.Scatter3d(
    x       = df_all["SS"], 
    y       = df_all["DBI"], 
    z       = df_all["CHI"], mode = "markers", 
    marker  = dict(size = 6, color = datamapplot_palette[2], opacity = 0.7),
    name    = "Trials", hoverinfo = "text", text = df_all["n_neighbors"]))

# Highlight best trial with a diamond marker
df_best = df_optuna[df_optuna["Best"]]

fig.add_trace(go.Scatter3d(
    x       = df_best["SS"], 
    y       = df_best["DBI"], 
    z       = df_best["CHI"], 
    mode    = "markers+text", 
    marker  = dict(
        size    = 8, 
        color   = datamapplot_palette[10], 
        symbol  = "diamond", 
        opacity = 1), 
    name    = "Best Trial"))

# Configure the layout of the 3D plot
fig.update_layout(
    # Title configuration
    title = dict(
        text = "3D Pareto Front",
        font = dict(size = 16, family = "Times New Roman"),
        x    = 0.5, # Center title horizontally
        y    = 0.9), # Move title closer to the plot
    
    # Legend configuration
    legend = dict(
        x           = 0.6, # Left edge (0.0 = far left, 1.0 = far right)
        y           = 0.8,  # Top (1.0 = very top)
        xanchor     = 'left', 
        yanchor     = 'top', 
        bgcolor     = 'rgba(255, 255, 255, 0.8)', 
        bordercolor = 'black',
        borderwidth = 1,
        font        = dict(family = 'Times New Roman', size = 12)),
    
    # Overall plot elements configuration
    font    = dict(family = "Times New Roman", size = 10),
    width   = 500,
    height  = 400,
    # Axis configuration
    scene = dict(
        # x-axis configuration
        xaxis = dict(
            title           = "SS", 
            titlefont       = dict(size = 12, family = "Times New Roman"),
            tickfont        = dict(size = 12, family = "Times New Roman"), 
            showline        = True,
            linecolor       = "black", 
            linewidth       = 1, 
            backgroundcolor = "white",
            gridcolor       = "lightgrey", nticks = 10),
        # y-axis configuration
        yaxis = dict(
            title           = "DBI", 
            titlefont       = dict(size = 12, family = "Times New Roman"),
            tickfont        = dict(size = 12, family = "Times New Roman"), 
            showline        = True,
            linecolor       = "black", 
            linewidth       = 1, 
            backgroundcolor = "white",
            gridcolor       = "lightgrey", 
            nticks          = 10),
        # z-axis configuration
        zaxis = dict(
            title           = "CHI", 
            titlefont       = dict(size = 12, family = "Times New Roman"),
            tickfont        = dict(size = 12, family = "Times New Roman"), 
            showline        = True,
            linecolor       = "black", 
            linewidth       = 1, 
            backgroundcolor = "white",
            gridcolor       = "lightgrey", 
            nticks          = 5)),
    margin          = dict(l = 10, r = 10, t = 60, b = 40),  # Small margins
    plot_bgcolor    = "white",
    paper_bgcolor   = "white",
    showlegend      = True)
    
# Save the plot
fig.write_html(f"{VISUALS_DIR}/Optuna_3D_Pareto.html")
webbrowser.open(f"{VISUALS_DIR}/Optuna_3D_Pareto.html")
#%%============================================================================
# UMAP: Fitting UMAP model using best_params from Optuna & save checkpoint:
# =============================================================================
best_umap = umap.UMAP(
    n_neighbors  = best_params["umap_n_neighbors"],
    min_dist     = best_params["umap_min_dist"],
    n_components = 2,
    metric       = "cosine", 
    random_state = 3)

# Save checkpoint (Optuna UMAP)
try:              
    CONFIG['checkpoints']['umap_done'] = True
    joblib.dump({"df" : df,
                 "embeddings" : embeddings,
                 "docs" : docs,
                 "best_trial" : best_trial,
                 "sil_score" : sil_score,
                 "db_score" : db_score,
                 "ch_score" : ch_score,
                 "best_umap" : best_umap, 
                 "best_params" : best_params, 
                 "umap_embeddings" : umap_embeddings,
                 "config" : CONFIG}, CONFIG["save_file"])
    print("Dimensionality reduction (UMAP) complete!")

except Exception as e:
    print(f"Dimensionality reduction (UMAP) failed: {str(e)}")
# %%===========================================================================
# HDBSCAN: Fitting HDBSCAN model using UMAP embeddings & best_params
# =============================================================================
best_hdbscan = hdbscan.HDBSCAN(
    min_cluster_size          = best_params["hdbscan_min_cluster_size"],
    min_samples               = best_params["hdbscan_min_samples"],
    cluster_selection_epsilon = best_params["hdbscan_cluster_selection_epsilon"],
    metric                    = "euclidean")

# Save checkpoint (Optuna HDBSCAN)
try:
    CONFIG["checkpoints"]["hdbscan_done"] = True
    joblib.dump({
        "df": df,
        "embeddings" : embeddings,
        "docs" : docs,   
        "best_trial" : best_trial,
        "sil_score" : sil_score,
        "db_score" : db_score,
        "ch_score" : ch_score,
        "best_params" : best_params, 
        "best_umap" : best_umap, 
        "umap_embeddings" : umap_embeddings,
        "best_hdbscan" : best_hdbscan,
        "labels" : labels,
        "config" : CONFIG}, 
        CONFIG["save_file"])
    print("HDBSCAN complete!")

except Exception as e:
    print(f"HDBSCAN failed: {str(e)}")
# %%===========================================================================
# Split Clustered DataFrame into Individual Cluster DataFrames
# =============================================================================
# Creating cluster specific documentation for future use:
df["Cluster"] = labels
clustered_df = df.copy()
cluster_groups = clustered_df.groupby("Cluster")
cluster_dataframes = {}

for cluster_id, group_data in cluster_groups:
    if cluster_id == -1:
        continue  # Skip noise points
        
    cluster_name = f"Cluster_{cluster_id}"
    cluster_dataframes[cluster_name] = group_data.reset_index(drop = True)

# Save to CSV
save_dir = f"{CLUSTERS_DIR}/Cluster_CSVs/"
os.makedirs(save_dir, exist_ok = True)

for cluster_name, df_cluster in cluster_dataframes.items():
     df_cluster.to_csv(f"{save_dir}{cluster_name}_Documents.csv", index = False)
     print(f"Saved {cluster_name} with {len(df_cluster)} documents.")
# %%===========================================================================
# Visualization of HDDBSCAN Clusters
# =============================================================================
def visualize_hdbscan_clusters(umap_embeddings, labels, save_path):
    """
    Generate & save a 2D scatter plot of HDBSCAN clusters 
    based on UMAP embeddings.
    
    Parameters
    ----------
        umap_embeddings : array-like, shape (n_samples, n_features)
            UMAP embeddings. If not 2D, a 2nd reduction is applied.
        labels : array-like, shape (n_samples,)
            Cluster labels from HDBSCAN (noise is labeled as -1)
        save_path : str
            File path where the .png of the plot will be saved.
    
    Returns
    -------
    None
        Saves the figure to 'save_path' & closes the plot.
    """     
    # Create new figure
    plt.figure(figsize = (8,6))
    # Extract unique labels & define color palette
    unique_labels = set(labels)
    colors = sns.color_palette(datamapplot_palette) # For readability
    # If embeddings != 2D, reduce them to 2D for plotting
    if umap_embeddings.shape[1] != 2:
        umap2d = umap.UMAP(
            n_neighbors     = best_params["umap_n_neighbors"],
            min_dist        = best_params["umap_min_dist"],
            n_components    = 2,
            metric          = "cosine",
            random_state    = 3)
 
        umap2d_embeddings = umap2d.fit_transform(umap_embeddings)
    else:
        umap2d_embeddings = umap_embeddings
        
    # Plot each cluster
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points in gray with low opacity
            plt.scatter(
                umap2d_embeddings[labels == label, 0],
                umap2d_embeddings[labels == label, 1],
                c     = "gray", 
                alpha = 0.05, 
                s     = 10, 
                label = "Noise")
        else:
            # Actual clusters in distinct colors
            plt.scatter(
                umap2d_embeddings[labels == label, 0],
                umap2d_embeddings[labels == label, 1],
                c     = [color], 
                alpha = 1.0, 
                s     = 20, 
                label = f"Cluster {label}")
            
    # Set titles & axis labels
    plt.title("HDBSCAN Clustering Visualization", fontsize = 14)
    plt.xlabel("UMAP Dimension 1", fontsize = 12)
    plt.ylabel("UMAP Dimension 2", fontsize = 12)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = "upper left", fontsize = 10)
    plt.tight_layout()
    
    # Compute axis limits based on percentiles to exclude outliers
    x_min, x_max = np.percentile(umap2d_embeddings[:, 0], [2, 98])
    y_min, y_max = np.percentile(umap2d_embeddings[:, 1], [2, 98])
    plt.xlim(x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min))
    plt.ylim(y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min))
    
    # Save the figure & close
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"Cluster visualization saved to {save_path}")

# Call the visualization function after HDBSCAN clustering
visualize_hdbscan_clusters(
    umap_embeddings, 
    labels, 
    save_path = f"{VISUALS_DIR}/HDBSCAN_Clustering.png")
#%%============================================================================
# BERTopic
# =============================================================================
# Parameters (for consistency)
ngram_range = (2,3)    # Bigrams & trigrams
min_df = 0.05          # Ignore terms that appear in less than 5% of documents
max_df = 0.5           # Ignore terms that appear in more than 50% of documents
n_words = 20           # Top 20 keywords from KeyBERT
n_repr_docs = 100      # 100 representative documents
token_pattern = r"(?u)\b[a-zA-Z]{5,}\b"

# Group documents by cluster
clustered_docs = pd.DataFrame({"doc": docs, "cluster": labels})
clustered_docs = clustered_docs[clustered_docs["cluster"] != -1]  # Exclude noise
# Concatenate documents per cluster
docs_per_cluster = clustered_docs.groupby("cluster")["doc"].apply(" ".join).reset_index()

# CountVectorizer
vectorizer_model = CountVectorizer( 
    stop_words    = list(stopwords_set),
    ngram_range   = ngram_range,
    min_df        = min_df,
    max_df        = max_df,
    token_pattern = token_pattern)
# Fit CountVectorizer
X = vectorizer_model.fit_transform(docs_per_cluster["doc"])

# c-TF-IDF Vectorizer
ctfidf_model = ClassTfidfTransformer()
# Fit c-TF-IDF Vectorizer
ctfidf_matrix = ctfidf_model.fit_transform(X)

# Extract top words per cluster
terms = vectorizer_model.get_feature_names_out()
top_words_per_cluster = {}
for i, cluster_id in enumerate(docs_per_cluster["cluster"]):
    row = ctfidf_matrix[i].toarray().flatten()
    top_indices = row.argsort()[::-1][:n_words]
    top_words = [terms[j] for j in top_indices]
    top_words_per_cluster[cluster_id] = top_words
    print(f"Cluster {cluster_id}: {top_words[:10]}")

# Representation models
kbert = KeyBERTInspired()
mmr   = MaximalMarginalRelevance(diversity = 0.99)
# Chain model
representation_models = [mmr, kbert]
    
# BERTopic
topic_model = BERTopic(
    embedding_model      = SentenceTransformer('all-MiniLM-L6-v2'),
    umap_model           = best_umap,
    hdbscan_model        = best_hdbscan,
    representation_model = representation_models,
    vectorizer_model     = vectorizer_model,
    ctfidf_model         = ctfidf_model,
    verbose              = True)

# Fit Transform Topic Model
topics, probs = topic_model.fit_transform(docs, embeddings)

# Save checkpoint:BERTopic
try:              
    CONFIG["checkpoints"]["BERTopic_done"] = True
    joblib.dump({"df": df, 
                 "embeddings" : embeddings,
                 "docs" : docs,   
                 "best_trial" : best_trial,
                 "sil_score" : sil_score,
                 "db_score" : db_score,
                 "ch_score" : ch_score,
                 "best_params" : best_params, 
                 "best_umap" : best_umap, 
                 "umap_embeddings" : umap_embeddings,
                 "best_hdbscan" : best_hdbscan,
                 "labels" : labels,
                 "topic_model" : topic_model, 
                 "topics" : topics,
                 "probs" : probs,
                 "config" : CONFIG}, 
                CONFIG["save_file"])
    print("BERTopic complete!")

except Exception as e:
    print(f"BERTopic failed: {str(e)}")
#%%============================================================================
# Representative documents from BERTopic & Metadata Mapping
# =============================================================================
# Get topic embeddings from BERTopic model
topic_embeddings = topic_model.topic_embeddings_
# Calculate cosine similarity between documents and each topic centroid
distances = cosine_similarity(embeddings, topic_embeddings)
# For each topic, find top N closest documents to topic centroid 
representative_docs = {}
for topic_id in range(len(topic_embeddings)):
    if topic_id == -1:  # Exclude noise
        continue
    # Sort document indices by similarity to the topic, take top n_repr_docs
    closest_doc_indices = np.argsort(
        distances[:, topic_id])[-n_repr_docs:][::-1]
    # Map topic_id to list of representative document strings
    representative_docs[topic_id] = [docs[i] for i in closest_doc_indices]

# Check N number of representative documents were pulled for each topic
for topic_id, docs_list in representative_docs.items():
    print(f"Topic {topic_id}: {len(docs_list)} representative documents")
    
# Organize topics by document size and extract top topic IDs
topic_info = topic_model.get_topic_info()

# Select top 15 topics by document count (excluding noise, -1)
top_topics = topic_info[topic_info.Topic != -1].sort_values(
    "Count", ascending = False).head(15)

top_topic_ids = top_topics["Topic"].tolist()

# Build a map from each topic to its top 5 representative words from KeyBERT
topic_words_map = {}
for topic_id in top_topic_ids:
    # Retrieve the list of (word, score) tuples for the topic
    words_with_scores = topic_model.get_topic(topic_id)
    # Extract the top 5 words from the list
    top5_words = [word for word, _ in words_with_scores][:5]
    # Join the words (string)
    topic_words_map[topic_id] = ", ".join(top5_words)

# Get full topic-document assignment (docs = preprocessed abstracts)
doc_info = topic_model.get_document_info(docs)
# Store original indices so they can be merged back to raw df later
doc_info["Original_Index"] = doc_info.index

# Build list of top 100 documents per topic with metadata
rows = []
for topic_id in top_topic_ids:
    # Get total size of current cluster
    cluster_size = int(topic_info[topic_info.Topic == topic_id]["Count"].values[0])
    # Extract 100 docs with highest assignment probability for current topic
    top_docs = doc_info[doc_info.Topic == topic_id].sort_values(
        "Probability", ascending = False).head(100)
    # Loop over each selected document & extract its metadata
    for rank, (_, row) in enumerate(top_docs.iterrows(), start=1):
        index = row["Original_Index"]
        metadata = df.iloc[index] # Retrieve title, journal, etc. from original df
        rows.append({
            "Topic" : topic_id,
            "Cluster_Size" : cluster_size,
            "Rank" : rank,
            "Title" : metadata["Title"],
            "Journal" : metadata["Journal"],
            "Date" : metadata["Date"],
            "DOI" : metadata["DOI"],
            "Representative_Abstract" : metadata["Abstract"],
            "Representative_Words" : topic_words_map.get(topic_id, "")})

# Create the dataframe of top 100 per topic entries 
df_top100 = pd.DataFrame(rows)

# Load full embeddings csv to merge on DOI (most accurate merging option)
df_all_emb = pd.read_csv(f"{SBERT_DIR}/All_Embeddings.csv")

# Fix embeddings from string to NumPy arrays
if isinstance(df_all_emb["Embedding"].iloc[0], str):
    df_all_emb["Embedding"] = df_all_emb["Embedding"].apply(fix_embedding_format)

# Primary merge on DOI to grab embeddings
merged  = df_top100.merge(df_all, on = "DOI", how = "left", 
                          suffixes = ("", "_full"))
# Find any rows that failed to merge
missing = merged[merged["Embedding"].isnull()]

# If any are missing, perform secondary merge using Title + Abstract as fallback
if not missing.empty:
    print(f"\{len(missing)} unmatched on DOI. Fallback to Title + Abstract...")

    # Build a lowercased "Join Key" in both dataframes
    df_top100["JoinKey"] = (df_top100["Title"].str.strip().str.lower() +
                            df_top100["Representative_Abstract"].str.strip().str.lower())
    df_all["JoinKey"] = (df_all["Title"].str.strip().str.lower() +
                         df_all["Abstract"].str.strip().str.lower())

    # Merge only unmatched entries again
    fallback_merge = df_top100.merge(df_all, on = "JoinKey", how = "left", 
                                     suffixes = ("", "_full"))
    # Grab newly matched rows & update main merged dataframe
    fallback_matched = fallback_merge[~fallback_merge["Embedding"].isnull()]
    merged.update(fallback_matched)

# Drop unmatched rows to ensure output is only valid top 100 matches
df_final = merged.dropna(subset=["Embedding"]).copy()
print(f"Final Top 100 Merged Set: {len(df_final)} entries (out of {len(df_top100)})")

# Save output
output_path = f"{MODELS_DIR}/Merged_Top100_With_Metadata.csv"
df_final.to_csv(output_path, index=False)
print(f"Output written to: {output_path}")
# =============================================================================
# Organize Abstracts with Metadata, BERTopic Topic ID, and Top 5 Keywords
# =============================================================================
# Add the BERTopic topic IDs to the metadata DataFrame
df["Topic_ID"] = topics
# Create a mapping from each topic ID to its top 5 keywords.
topic_keywords = {}

for topic in set(topics):
    if topic == -1: # Noise = Topic -1
        topic_keywords[topic] = "Noise"
    else:
        # Retrieve the topic terms (each tuple is (term, score))
        topic_terms = topic_model.get_topic(topic)
        # Extract  top 5 terms and join them as a comma sep str
        top5_terms = [term for term, score in topic_terms[:5]]
        topic_keywords[topic] = ", ".join(top5_terms)
        topic_keybert_terms = topic_model.get_topic(topic)
        top5_terms = [term for term, score in topic_terms[:5]]
        
# Map the top 5 keywords to each document based on Topic_ID
df["Top_5_Keywords"] = df["Topic_ID"].map(topic_keywords)

# Write the consolidated DataFrame to a CSV file.
output_csv_path = f"{MODELS_DIR}/AllAbstracts_BERTopic.csv"
df.to_csv(output_csv_path, index=False)
print(f"CSV file with abstracts and topic metadata saved to {output_csv_path}")
#%%============================================================================
#  Visualizations
# =============================================================================
# Custom labels for WordCloud & DataMapPlot
user_labels = {-1: "Noise", 
               0: "Waste Reduction",
               1: "Microplastics",
               2: "Adsorption & Membranes",
               3: "Recycled Concrete Aggregate",
               4: "Biodegradable Packaging",
               5: "Green Building Design",
               6: "Materials Properties",
               7: "Plastic Pyrolysis",
               8: "Additive Manufacturing",
               9: "Green Polymer Composites",
               10: "Wastewater",
               11: "Agriculture",
               12: "Battery Recycling",
               13: "Biorefining",
               14: "Modified Asphalt Mixtures"}
#%%============================================================================
# BERTopic Bar Chart
# =============================================================================
def visualize_topic_barchart(topic_model, docs, n_gram_range, vectorizer_model, 
                             top_n_topics, n_words, save_path):
    
    topic_model.update_topics(
        docs, 
        n_gram_range     = n_gram_range, 
        top_n_words      = n_words, 
        vectorizer_model = vectorizer_model)
    
    fig = topic_model.visualize_barchart(
        top_n_topics = top_n_topics, 
        n_words      = n_words)

    fig.update_layout(
        font = dict(
            family  = "Times New Roman", 
            size    = 12, 
            color   = "black"), 
        
        margin   = dict(l = 250, r = 20, t = 100, b = 100),
        width    = 1400,
        height   = 400 + (50 * len(representative_docs)),
        autosize = False,
        title_x  = 0.5,  # Center title
        title_y  = 1,
        title_xanchor = "center",
        hoverlabel    = dict(bgcolor = "white", font_size = 12))

    fig.update_yaxes(
        automargin     = True,
        title_standoff = 50,
        tickfont       = dict(size = 10))

    fig.update_xaxes(title_standoff = 20, automargin = True)
    plot(fig)
    fig.write_html(save_path)
    
# BERTopic Bar Chart Execution
visualize_topic_barchart(
    topic_model      = topic_model, 
    docs             = docs, 
    n_gram_range     = ngram_range, 
    vectorizer_model = vectorizer_model, 
    top_n_topics     = len(user_labels) + 1, 
    n_words          = 10,
    save_path        = f'{VISUALS_DIR}/BERTopic_BarChart.html')
#%%============================================================================
# BERTopic Visualize Documents
# =============================================================================
# Run the visualization with the original embeddings
def visualize_docs(topic_model, docs, embeddings, save_path):
    fig = topic_model.visualize_documents(docs, embeddings)
    fig.update_layout(font = (dict(
        family = "Times New Roman", size = 12, color = "black")))
    plot(fig)
    fig.write_html(save_path)

# Visualize documents execution
save_path = f"{VISUALS_DIR}/BERTopic_VisualizeDocs.html"
visualize_docs(topic_model, docs, embeddings, save_path = save_path)
#%%============================================================================
# BERTopic WordCloud
# =============================================================================
def create_wordcloud(topic_model, save_wordcloud = f"{VISUALS_DIR}/WordClouds/"):
    # Update BERTopic to contain 100 words
    topic_model.update_topics(docs, top_n_words = 100, 
                              vectorizer_model = vectorizer_model)
    
    for topic in topic_model.get_topics():
        # Top 100 words with c-TF-IDF scores
        topic_words = topic_model.get_topic(topic)
        text = {word: value for word, value in topic_words}
        wc = WordCloud(
            background_color = "white",
            colormap         = "Clepticus_parrae", # pypalettes
            max_words        = 100,
            width            = 1600,
            height           = 800)
        wc.generate_from_frequencies(text)
        
        label = user_labels.get(topic, f"Topic_{topic}")
        safe_label = label.replace(" ", "_").replace("/", "_")
        plt.figure(figsize=(20,10))
        plt.imshow(wc, interpolation = "bilinear")
        plt.axis("off")
        plt.title(f"{label}", fontsize = 48, pad = 20)
        plt.savefig(
            f"{save_wordcloud}WordCloudTopic{topic}_{safe_label}.png",
            dpi = 600, 
            bbox_inches = 'tight') 
            
        plt.close()

# WordCloud execution
create_wordcloud(topic_model)
#%%============================================================================
# DataMapPlot from BERTopic & with customizations
# =============================================================================
custom_topic_labels = {}
for topic_id in sorted(topic_model.get_topics().keys()):
    if topic_id == -1:
        custom_topic_labels[topic_id] = "Unlabeled"
    else:
        # Title line
        title       = user_labels.get(topic_id, f"Topic {topic_id}")
        title_bold  = title.replace("_", r"\_").replace(" ", r"\;")
        line1       = rf"$\bf{{{title_bold}}}$"

        # Keyword line
        keywords = [kw.replace("_", r"\_") 
                    for kw, _ in topic_model.get_topic(topic_id)][:5]
        
        line2 = ", ".join(keywords)

        # Final label
        final_label = f"{line1}\n{line2}"
        custom_topic_labels[topic_id] = final_label

# Assign the custom labels 
topic_model.custom_labels_ = custom_topic_labels

# Error handling
try: 
    import datamapplot 
    from matplotlib.figure import Figure 
except ImportError: 
    warn("Data map plotting is unavailable unless datamapplot is installed.") 
# =============================================================================
# DataMapPlot Function from BERTopic documentation    
# =============================================================================
    # Create a dummy figure type for typing
    class Figure(object):
        pass
    
def visualize_document_datamap(
    topic_model,
    docs: List[str],
    topics: List[int]               = None,
    embeddings: np.ndarray          = None,
    reduced_embeddings: np.ndarray  = None,
    custom_labels: Union[bool, str] = False,
    title: str = "Plastics - Documents and Topics",
    sub_title: Union[str, None]     = None,
    width: int       = 1000,
    height: int      = 1000,
    dpi              = 1200,
    point_alpha      = 1.0,
    label_alpha      = 1.0,
    point_size       = 6,
    label_font_size  = 8,
    background_color = "white",
    font_color       = "black",
    palette          = "tab10",
    **datamap_kwds
) -> Figure:
    """Visualize documents and their topics in 2D as a static plot for publication using
    DataMapPlot.

    Arguments:
        topic_model: A fitted BERTopic instance.
        docs: Documents you used when calling either `fit` or `fit_transform`
        topics: A selection of topics to visualize.
        embeddings:  The embeddings of all documents in `docs`.
        reduced_embeddings: 2D reduced embeddings of all documents in `docs`.
        custom_labels: If bool, whether to use custom topic labels that were
                       defined using `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, 
                       e.g., "Aspect1".
        title: Title of the plot.
        sub_title: Sub-title of the plot.
        width: The width of the figure.
        height: The height of the figure.
        **datamap_kwds: All further keyword args will be passed on to
                        DataMapPlot's `create_plot` function. See the 
                        DataMapPlot documentation for more details.

    Returns:
        figure: A Matplotlib Figure object.
    """
    
    topic_per_doc = topic_model.topics_
    df = pd.DataFrame({"topic": np.array(topic_per_doc)})
    df["doc"] = docs

    # Extract embeddings if not already done
    if embeddings is None and reduced_embeddings is None:
        embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
    else:
        embeddings_to_reduce = embeddings

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = umap.UMAP(
            n_neighbors  = best_params['umap_n_neighbors'],
            min_dist     = best_params['umap_min_dist'],
            n_components = 2, 
            metric       = 'cosine', 
            random_state = 3
            ).fit(embeddings_to_reduce)   
        
        embeddings_2d = umap_model.embedding_
    else:
        embeddings_2d = reduced_embeddings
        
    unique_topics = set(topic_per_doc)

    if isinstance(custom_labels, str):
        # (Existing branch for aspect-based custom labels if provided as a string)
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] 
                 for topic in unique_topics]
        
        names = [" ".join([label[0] for label in labels[:10]]) 
                 for labels in names]
        
        names = [label if len(label) < 30 else label[:20] + "..." 
                 for label in names]
        
    elif topic_model.custom_labels_ is not None and custom_labels:
        # Use our custom labels dictionary directly
        names = []
        for topic in unique_topics:
            if topic == -1:
                names.append("Unlabelled")
            else:
                # Use the label already formatted in topic_model.custom_labels_
                names.append(topic_model.custom_labels_.get(topic, f"Topic-{topic}"))
    else:
        # Fallback: default topic names using the top 5 keywords
        names = [
            f"Topic-{topic}: " + " ".join(
                [word for word, value in topic_model.get_topic(topic)][:5]) 
            for topic in unique_topics
            ]

    topic_name_mapping = {
        topic_num: topic_name 
        for topic_num, topic_name in zip(unique_topics, names)
        }
    
    topic_name_mapping[-1] = "Unlabelled"

    if topics is not None:
        selected_topics = set(topics)
        for topic_num in topic_name_mapping:
            if topic_num not in selected_topics:
                topic_name_mapping[topic_num] = "Unlabelled"

    named_topic_per_doc = pd.Series(topic_per_doc).map(topic_name_mapping).values
    
    # Color map for topics (excluding noise)
    unique_named_topics = set(named_topic_per_doc)
    
    # Custom color palette from mycolors.py
    CUSTOM_COLORS = datamapplot_palette

    # Skip gray entirely
    color_idx = 0
    label_color_map = {}
    for label in sorted(unique_named_topics):
        if label == "Unlabelled":
            continue
        label_color_map[label] = CUSTOM_COLORS[color_idx % len(CUSTOM_COLORS)]
        color_idx += 1
    
    # Hex string with alpha isn't supported, so use gray (apply alpha post hoc)
    noise_color = "#d3d3d3"  # Very light gray

    figure, axes = datamapplot.create_plot(
        embeddings_2d,
        named_topic_per_doc,
        add_glow        = False,
        figsize         = (width / 100, height / 100),
        dpi             = dpi,
        title           = title,
        sub_title       = sub_title,
        label_color_map = label_color_map,
        noise_color     = noise_color,
        point_size      = point_size,
        label_font_size = label_font_size,
        **datamap_kwds
    )
    
    # Post-process to apply alpha = 0.1 to noise points (topic -1)
    topic_array = np.array(topic_per_doc)
    
    for ax in figure.axes:
        for artist in ax.collections:
            if isinstance(artist, mc.PathCollection):
                offsets = artist.get_offsets()
                facecolors = artist.get_facecolors()
    
                if len(offsets) == len(topic_array):
                    for i, topic in enumerate(topic_array):
                        if topic == -1:
                            facecolors[i][3] = 0.03  # Set alpha to 0.1
                        else:
                            facecolors[i][3] = 1.0  # Ensure topic points are opaque
                    artist.set_facecolors(facecolors)
    return figure
#%%============================================================================
# DataMapPlot Execution
# =============================================================================
save_path   = f"{VISUALS_DIR}/DataMapPlot.png"
vis_umap    = umap.UMAP(n_components = 2, random_state = 3)
reduced_embeddings = vis_umap.fit_transform(umap_embeddings)
topics      = topic_model.topics_

figure = visualize_document_datamap(
    topic_model,
    docs,
    reduced_embeddings = reduced_embeddings,
    custom_labels=True,
    title = "Plastic Abstracts")

figure.savefig(save_path, dpi = 1200, bbox_inches = "tight")


#%%============================================================================
# Temporal Analysis Plots
# =============================================================================
# Directories
ALL_CLUSTERS_PATH = f"{CLUSTER_CSV_DIR}/All_Clusters.xlsx"
df = pd.read_excel(ALL_CLUSTERS_PATH)
# =============================================================================
# Load data from NLP_Study_Final.pkl file (if needed)
# =============================================================================
data = joblib.load(r"D:/MS_Thesis/Scripts/NLP_Study_Final.pkl")
topic_model  = data['topic_model']     # BERTopic topic model
embeddings   = data['embeddings']      # SentenceBERT embeddings
docs         = data["docs"]            # Abstracts
best_umap    = data["best_umap"]       # UMAP fit with Optuna
best_hdbscan = data["best_hdbscan"]    # HDBSCAN fit with Optuna
# CountVectorizer as defined in NLP.py
vectorizer  = CountVectorizer(        
    ngram_range     = (2,3), 
    min_df          = 0.05, 
    max_df          = 0.5, 
    token_pattern   = r"(?u)\b[a-zA-Z]{5,}\b")
# c-TF-IDF
ctfidf  = ClassTfidfTransformer() 
# Chain model (MMR & KeyBERT)
rep_models = [MaximalMarginalRelevance(diversity = 0.99), 
              KeyBERTInspired()]
# Fit BERTopic Topic Model from NLP_study.pkl
topics, probs = topic_model.fit_transform(docs, embeddings)

# Update the topic representations to 100 words
topic_model.update_topics(docs, top_n_words = 100, 
                          vectorizer_model = vectorizer)
#%%=========================================================================
# Extract KeyWords used to construct WordCloud
# =============================================================================
all_kw = {}
for topic_id in topic_model.get_topic_info().Topic:
    if topic_id == -1:
        continue
    # A list of 100 (word, score) tuples
    terms_scores = topic_model.get_topic(topic_id) 
    words  = [w for w, _ in terms_scores[:100]]
    scores = [s for _, s in terms_scores[:100]]
    all_kw[topic_id] = dict(zip(words, scores))
#%%============================================================================
# Temporal Analysis for ALL terms for ALL documents
# =============================================================================
df = pd.read_excel(ALL_CLUSTERS_PATH)
lower_yr = 2015
upper_yr = 2025
df = df[df["Year"].between(lower_yr, upper_yr)]
years = list(range(lower_yr, (upper_yr + 1)))

# All unique topic keywords (from all_kw)
all_terms = set()
for tid, wordscores in all_kw.items():
    if tid == -1:
        continue
    all_terms.update(wordscores.keys())

# Rank keywords by total frequency in entire corpus
all_text = df["Abstract"].str.cat(sep = " ").lower()
term_freq = {term: all_text.count(term) for term in all_terms}

# Select top N most frequent terms globally
top_n = 20
top_terms = sorted(term_freq, key = term_freq.get, reverse = True)[:top_n]

# Calculate frequency per year for top terms
term_counts_by_year = {term: [] for term in top_terms}

for yr in years:
    text = df[df["Year"] == yr]["Abstract"].str.cat(sep=" ").lower()
    for term in top_terms:
        term_counts_by_year[term].append(text.count(term))

# Plot
plt.figure(figsize=(8,5))
for color, term in zip(temporal_palette, top_terms):
    plt.plot(years, term_counts_by_year[term], label=term, color=color, linewidth=1)


plt.xticks(np.arange(lower_yr, (upper_yr + 1)))
plt.xlim(lower_yr, upper_yr)
plt.xlabel("Publication Year", fontsize = 12)
plt.ylabel("Raw Term Count", fontsize = 12)
plt.title(f"Top 20 Globally Frequent BERTopic Keywords ({lower_yr}‒{upper_yr})", 
          fontsize = 14)
plt.legend(bbox_to_anchor = (1.02,1), 
           loc = "upper left", 
           frameon = False, 
           fontsize = "small")
plt.tight_layout()
#plt.show()
plt.savefig(f"{VISUALS_DIR}/Top_Global_Term_Trends_{lower_yr}_{upper_yr}.png", 
            dpi = 300, 
            bbox_inches = "tight")
#%%============================================================================
# Temporal Analysis for Microplastics Cluster (Cluster 1)
# plot_cluster_keyword_trends execution
# =============================================================================
df = pd.read_excel(ALL_CLUSTERS_PATH)
plot_cluster_keyword_trends(
    df = df,
    cluster_id = 1,
    all_kw = all_kw,
    cluster_name = "Microplastics",
    lower_yr = 2020,
    upper_yr = 2025,
    temporal_palette = temporal_palette,
    VISUALS_DIR = VISUALS_DIR,
    top_n = 20,
    tick_interval = 1)
#%%============================================================================
# Temporal Analysis for Biodegradable Packaging Cluster (Cluster 4) 
# plot_cluster_keyword_trends execution
# =============================================================================
df = pd.read_excel(ALL_CLUSTERS_PATH)
plot_cluster_keyword_trends(
    df = df,
    cluster_id = 4,
    all_kw = all_kw,
    cluster_name = "Biodegradable Packaging",
    lower_yr = 2020,
    upper_yr = 2025,
    temporal_palette = temporal_palette,
    VISUALS_DIR = VISUALS_DIR,
    top_n = 20,
    tick_interval = 1)
#%%============================================================================
# Temporal Analysis for Plastic Pyrolysis Cluster (Cluster 7)
# plot_cluster_keyword_trends execution
# =============================================================================
df = pd.read_excel(ALL_CLUSTERS_PATH)
plot_cluster_keyword_trends(
    df = df,
    cluster_id = 7,
    all_kw = all_kw,
    cluster_name = "Plastic Pyrolysis",
    lower_yr = 2020,
    upper_yr = 2025,
    temporal_palette = temporal_palette,
    VISUALS_DIR = VISUALS_DIR,
    top_n = 20,
    tick_interval = 1)