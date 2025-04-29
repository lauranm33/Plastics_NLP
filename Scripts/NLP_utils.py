import concurrent.futures
import hdbscan, optuna, umap
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
import re
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                             calinski_harabasz_score)
import sys
#%%============================================================================
# Custom color palettes for plotting
# =============================================================================
"""
Color palettes used for thesis figures. 
"""
gem_palette = [
    "#6A0DAD",  # Amethyst (purple)
    "#00796B",  # Emerald (deep green)
    "#1E88E5",  # Sapphire (royal blue)
    "#C2185B",  # Ruby (crimson pink)
    "#FFC107",  # Topaz (golden yellow)
    "#8D6E63",  # Smoky Quartz (earthy taupe)
    "#00ACC1",  # Aquamarine (teal)
    "#F4511E"]  # Carnelian (burnt orange)

datamapplot_palette = [
    "#3CBCFC", # light blue
    "#0058F8", # dark blue
    "#9878F8", # medium orchid
    "#6844FC", # deep purple
    "#D800CC", # hot pink
    "#E40058", # bright red
    "#A80020", # maroon
    "#DD8709", # orange
    "#00B800", # green
    "#007800", # forest green
    "#6888FC", # grey blue
    "#B8F818", # lime green
    "#F8B800", # light orange
    "#009A96", # teal
    "#F878F8", # pink
    "#a9a9a9"] # dark gray

temporal_palette = ['#5E0D2E', '#810E1F', '#A51A0D', '#CB4B0B', '#F49106', 
                   '#18D80E', '#23A509', '#1C8307', '#146105', '#0E816D', 
                   '#0D98A5', '#0B8BCB', '#0669F4', '#1F44FF', '#0600AD', 
                   '#2E0A8A', '#660DA5', '#AB0BCB', '#F406E0', '#FF5CFF']

def get_palette(name = "gem"):
    if name == "gem":
        return gem_palette
    elif name == "datamapplot":
        return datamapplot_palette
    elif name == "temporal":
        return temporal_palette
    else:
        raise ValueError(f"Unknown palette name: {name}")
# =============================================================================
# For SentenceTransformer_Embeddings.py, shown in order of use
# =============================================================================
def process_chunk(model, output_folder, chunk, i):
    """
    Encode a chunk of documents with a SentenceTransformer model & save to .CSV

    Parameters
    ----------
    model : sentence_transformers.SentenceTransformer
        Preloaded SBERT model used to generate embeddings.
    output_folder : str
        Path to the directory where chunk CSVs will be written.
    chunk : pandas.DataFrame
        DataFrame containing at least an "Abstract" column of texts to encode.
    i : int
        The index of this chunk (used to construct the output filename).

    Returns
    -------
    None.
        Writes out a file named 'Embeddings_Chunk_{i}.csv' into 
        'output_folder', but does not return a value.
    """
    
    os.makedirs(output_folder, exist_ok = True)
    
    all_embeddings = model.encode(
        chunk["Abstract"].tolist(), 
        batch_size = 64, 
        show_progress_bar = True)
    
    # Attach embeddings & save to a chunked csv file
    chunk["Embedding"] = list(all_embeddings)
    output_file = os.path.join(output_folder, f"Embeddings_Chunk_{i}.csv")
    chunk.to_csv(output_file, index = False)
    
# =============================================================================
    
def concatenate_embedding_chunks(output_folder, SBERT_DIR):
    """
    Concatenate all SBERT embedding chunk CSVs in 'output_folder' into a 
    single DataFrame and save it as All_Embeddings.csv under SBERT_DIR.

    Parameters
    ----------
    output_folder : str
        Directory containing chunk files named Embeddings_Chunk_{i}.csv
    SBERT_DIR : str
        Base SBERT directory used in the output filename.

    Returns
    -------
    pandas.DataFrame
        The concatenated DataFrame of all chunks.
    """
    # List all the chunked CSV files
    chunk_files = [
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
        if f.startswith("Embeddings_Chunk_") and f.endswith(".csv")]

    # Sort files numerically to maintain order
    chunk_files.sort(key = lambda x: int(x.split("_")[-1].split(".")[0]))

    # Read and concatenate all chunk files
    df_all = pd.concat((pd.read_csv(f) for f in chunk_files), 
                       ignore_index = True)

    # Save the concatenated file
    output_file = os.path.join(output_folder, f"{SBERT_DIR}/All_Embeddings.csv")
    df_all.reset_index(drop=True)
    df_all.to_csv(output_file, index=False)
    print(f"All embedding chunks have been concatenated and saved to {output_file}")

    return df_all

# =============================================================================

def fix_embedding_format(embedding_str):
    """ 
    Convert string embedding (e.g. "[0.1 0.2 ...]") into Numpy array of floats.
    
    Parameters
    ----------
    embedding_str : str or array-like
        Either a string of numbers in square-brackets or a numeric sequence.
        
    Returns
    -------
    numpy.ndarray
        1D array of floats representing the document embedding. 
        If input is not a string, returns an array of 0s of length 768.
    
    Example Usage
    -------------
    >>> df['Embedding'] = df['Embedding'].apply(fix_embedding_format)
    >>> type(df.loc[0, 'Embedding'])
    <class 'numpy.ndarray'>
    """
    numbers = re.findall(r"[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d*\.\d+|\d+", 
                         embedding_str)
    
    return np.array(numbers, dtype = float)

# =============================================================================

def compute_top_k_similarities(embeddings, top_k = 10):
    """
    Compute top-k most similar embeddings for each embedding in the dataset.

    Parameters
    ----------
    embeddings : array-like, shape (N, D)
        Array of N embedding vectors of dimension D.
    top_k : int, default = 10
        Number of top similar embeddings to identify for each vector.

    Returns
    -------
    top_k_indices : ndarray of shape (N, top_k), dtype = int
        Indices of the top-k most similar embeddings for each entry.
    top_k_values : ndarray of shape (N, top_k), dtype = float32
        Cosine similarity scores corresponding to the top-k indices.
        
    """
    
    # Number of embeddings 
    N = embeddings.shape[0]
    # Initiate N by top_k shaped arrays for indices and similarity values
    top_k_indices = np.zeros((N, top_k), dtype = int)
    top_k_values = np.zeros((N, top_k), dtype = np.float32)
    
    # Compute L2 norms for each embedding once
    norms = np.linalg.norm(embeddings, axis = 1)
    
    # For each embedding, compute similarities and select top_k
    for i in range(N):
        # Dot product of all embeddings with embeddings[i]
        dot_products = embeddings @ embeddings[i]
        # Normalize to get cosine similarity
        similarity = dot_products / (norms * norms[i] + 1e-10)
        # Exclude self-match by setting its similarity to itself -inf
        similarity[i] = -np.inf
        # Unsorted top_k indices
        top_k_idx = np.argpartition(-similarity, top_k)[:top_k]
        # Sort top_k indices by descending similarity
        top_k_idx_sorted = top_k_idx[np.argsort(-similarity[top_k_idx])]
        # Store sorted indices w/ their similarity values
        top_k_indices[i] = top_k_idx_sorted
        top_k_values[i] = similarity[top_k_idx_sorted]

    return top_k_indices, top_k_values

# =============================================================================

def save_sbert_performance(top_k_indices_chunked, 
                           top_k_values_chunked, 
                           sbert_performance_path):
    """
    Save SBERT top-k nearest-neighbor indices & similarity scores to .CSV

    Parameters
    ----------
    top_k_indices_chunked : array-like of shape (N, k)
        For each of N query chunks, the indices of top-k most similar documents.
    top_k_values_chunked : array-like of shape (N, k)
        For each of N query chunks, the similarity scores corresponding 
        to the top-k indices.
    sbert_performance_path : str
        path/(including filename) where .csv will be written.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
            - "Abstract Index" : chunk/query index (0 through N-1)
            - "Top_i_Index" : the document index of i-th neighbor
            - "Top_i_Similarity": the similarity score of the i-th neighbor
        One row per query chunk.
        
    """
    
    # Number of queries (N) and number of neighbors (k)
    N, top_k = top_k_indices_chunked.shape
    data = []
    
    # Build a list of rows, each with: 
    # [query_index, idx1, idx2, ..., score1, score2, ...]
    for i in range(N):
        row = ([i]      # Query/chunk index
               + list(top_k_indices_chunked[i])    # Append k neighbor indices
               + list(top_k_values_chunked[i]))    # Append k similarity scores
        data.append(row)
        
    # Define column names
    columns = (["Abstract Index"] 
               + [f"Top_{i+1}_Index" for i in range(top_k)] 
               + [f'Top_{i+1}_Similarity' for i in range(top_k)])
    
    # Create DataFrame & save to .csv
    df_perf = pd.DataFrame(data, columns = columns)
    df_perf.to_csv(sbert_performance_path, index = False)
    print(f"SBERT performance saved to {sbert_performance_path}")

    return df_perf

# =============================================================================
    
def plot_similarity_distribution(top_k_values, sim_path = None):
    """
    Plot & (optionally) save histogram of cosine similarity scores 
    for embeddings.

    Parameters
    ----------
    top_k_values : array-like of shape (N, k)
        Similarity scores for the top-k neighbors, across N items.
    sim_path : str or None, optional
        File path to save the histogram PNG. 
        If None, the plot is only shown. The default is None.

    Returns
    -------
    None.
    
    """
    
    # Create a new figure
    plt.figure(figsize = (5, 3))
    # Plot flattened 1D (from 2D) array of similarity scores with 50 bins,
    # custom color, & transparency
    plt.hist(top_k_values.flatten(), 
             bins = 50, 
             color = "#A80020", # Dark red
             alpha = 0.7)
    
    # Label axes & title
    plt.xlabel('Cosine Similarity Score', fontsize = 10)
    plt.ylabel('Frequency', fontsize = 10)
    plt.title('Distribution of Top-K Similarities', fontsize = 12)
    
    # Save if a file path is provided
    if sim_path:
        plt.savefig(sim_path, dpi = 300, bbox_inches = 'tight')
        
    # Display the plot
    plt.show()
    
# =============================================================================

def plot_gaussian_similarity_distribution(top_k_values, gauss_path = None): 
    """
    Plot a smoothed (Gaussian + spline technique) distribution of 
    cosine similarities for embeddings.

    Parameters
    ----------
    top_k_values :  array-like of shape (N, k)
        Similarity scores for the top-k neighbors, across N items.
        DESCRIPTION.
    gauss_path : str or None, optional
        File path to save the smoothed distribution PNG. 
        If None, the plot is only shown. The default is None.

    Returns
    -------
    None.

    """
    
    # Create new figure
    plt.figure(figsize = (5, 3))
    
    # Flatten similarity scores & compute raw histogram counts & 
    # bin edges over a fixed range
    counts, bins = np.histogram(top_k_values.flatten(),
                                bins = 100, 
                                range = (0.2, 1.0))
    
    # Apply Gaussian smoothing technique to raw counts
    smoothed_counts = gaussian_filter1d(counts, sigma=2)
    
    # Calculate bin centers for interpolation
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Generate finer grid for the smooth curve
    x_new = np.linspace(bin_centers.min(), bin_centers.max(), 300)
    # Fit a cubic spline to smoothed counts for smoother curve
    spline = make_interp_spline(bin_centers, smoothed_counts, k = 3)
    y_smooth = spline(x_new)
    
    # Plotting
    # Fill under curve & plot its outline
    plt.fill_between(x_new, y_smooth, color = "#A80020", alpha = 0.9)
    plt.plot(x_new, y_smooth, color = "#A80020", linewidth = 2)
    # Configure tick marks & font sdizes
    plt.xticks(np.arange(0.2, 1.1, 0.2), fontsize = 10)
    plt.yticks(np.arange(0, 16000, 2500), fontsize = 10)
    plt.tick_params(axis = "both", which = "major", labelsize = 10)
    # Adjust y-axis to show full distribution & accomodate smoothed peak
    plt.ylim(0, max(counts) * 1.1)
    # Label axes & title
    plt.xlabel("Cosine Similarity Score", fontsize = 12)
    plt.ylabel("Frequency", fontsize = 12)
    plt.title("Smoothed Distribution of Top-K Similarities", fontsize = 14)
    
    # Save if a file path is provided
    if gauss_path:
        plt.savefig(gauss_path, dpi = 300, bbox_inches = "tight", 
                    facecolor = "white")
    # Display the plot
    plt.show()
#%%===========================================================================
# Utilities for BERTopic Pipeline Script: NLP.py
# =============================================================================

# Stopwords sourced from github, USPTO, technical-stopwords, and custom-built
# https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.txt

def preprocess_document(doc, stopwords_set):
    """
    Preprocess 1 document by lowercasing, removing special characters,
    tokenizing, lemmatizing, & filtering out stopwords & short lemmas.
    
    Parameters
    ----------
        doc : str
            Raw text of document (e.g., an abstract) to be preprocessed 
        stopwords_set : set of str
            Set of stopwords to remove from documents
            
        Returns
        -------  
        str
            A space separated string of lemmatized tokens w/ stopwords & tokens
            of length <= 2 rremoved.
            
    Example Usage
    -------------
    >>> from nltk.corpus import stopwords
    >>> from NLP import preprocess_document                       
    >>> stop = set(stopwords.words('english'))
    >>> text = "The quick, brown foxes jumped over 2 lazy dogs!"
    >>> preprocess_document(text, stop)
    'dog cute like playing'       # By default, all words are treated as nouns.
    """
    
    # Lowercase and remove special characters (keep spaces)
    doc = re.sub(r"[^A-Za-z0-9 ]+", "", doc.lower())
    
    # Tokenize, lemmatize, and filter
    lemmatizer = WordNetLemmatizer() 
    tokens = [lemmatizer.lemmatize(token) 
              for token in re.findall(r"\b\w+\b", doc)
              if token not in stopwords_set and len(token) > 2]
    
    return " ".join(tokens)

# =============================================================================

def preprocess_corpus(abstracts, stopwords_filepath):
    """
    Load a custom stopword list, merge it with NLTK’s English stopwords,
    and apply 'preprocess_document' to each abstract.

    Parameters
    ----------
    abstracts : list of str
        The raw documents (e.g. abstracts) to preprocess.
    stopwords_filepath : str
        Path to a newline-separated file of extra/custom stopwords.

    Returns
    -------
    docs : list of str
        Each abstract after preprocessing (lowercased, stripped, tokenized,
        lemmatized, filtered, and re-joined).
    stopwords_set : set of str
        The combined set of stopwords used for preprocessing.
    """
    # Load file’s stopwords
    with open(stopwords_filepath, "r", encoding = "utf-8") as f:
        custom = set(f.read().splitlines())

    # Merge with NLTK’s English list
    stopwords_set = custom.union(set(stopwords.words("english")))
    
    # Preprocess each document
    docs = [preprocess_document(abstract, stopwords_set) 
            for abstract in abstracts]

    return docs, stopwords_set

# =============================================================================

def objective(trial, embeddings):
    """ 
    Objective function for Optuna optimization of UMAP & HDBSCAN hyperparams.
    Intended for maximizing silhouette score (approach +1) & 
    Calinski Harabasz Index (approach +inf) & 
    minimizing Davies Bouldin Index (approach 0).
    
    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object, used to suggest hyperparameters.
    embeddings : array-like, shape (n_samples, n_features)
        Precomputed SentenceTransformer embeddings.
    
    Returns
    -------
    tuple of float
        (silhouette_score, -davies_bouldin_index, calinski_harabasz_index)
        - silhouette_score : maximized
        - davies_bouldin_index : minimized
        - calinski_harabasz_index : maximized
    """
# =============================================================================
# UMAP
# =============================================================================
    umap_params = {
        "n_neighbors": trial.suggest_int(
            "umap_n_neighbors", 25, 50),
        "min_dist": trial.suggest_float(
            "umap_min_dist", 0.05, 0.075),
        "n_components": 2}
    # Reduce dimensionality by fitting UMAP model to data
    umap_model = umap.UMAP(**umap_params, metric = "cosine", random_state = 3)
    umap_embeddings = umap_model.fit_transform(embeddings)
# =============================================================================
# HDBSCAN
# =============================================================================
    hdbscan_params = {
        "min_cluster_size": trial.suggest_int(
            "hdbscan_min_cluster_size", 400, 500),
        "min_samples": trial.suggest_int(
            "hdbscan_min_samples", 25, 50),
        "cluster_selection_epsilon": trial.suggest_float(
            "hdbscan_cluster_selection_epsilon", 0.05, 0.075)} 
    # Perform clustering
    hdbscan_model = hdbscan.HDBSCAN(**hdbscan_params, metric = "euclidean")
    labels = hdbscan_model.fit_predict(umap_embeddings)
# =============================================================================
# Calculate metrics, early convergence, scoring
# =============================================================================
    valid_labels = labels[labels != -1] # Removing noise 
    valid_embeddings = umap_embeddings[labels != -1]
    n_clusters = len(np.unique(valid_labels))
    
    # Penalize out of range clusters (< 15 clusters but > 5 clusters)
    if n_clusters < 5 or n_clusters > 15:
        raise optuna.TrialPruned() # Stop trial early
        
    if np.sum(labels == -1) / len(labels) > 0.5: # If very noisy, stop early
        raise optuna.TrialPruned()
        
    try:
        sil_score = silhouette_score(valid_embeddings, valid_labels)
        db_score = davies_bouldin_score(valid_embeddings, valid_labels)
        ch_score = calinski_harabasz_score(valid_embeddings, valid_labels)
    except:
        raise optuna.TrialPruned()
        
    return sil_score, db_score, ch_score

#%%============================================================================
# Temporal Analysis Plotting
# =============================================================================
def plot_cluster_keyword_trends(df, cluster_id, all_kw, cluster_name, lower_yr, 
                                upper_yr, temporal_palette, VISUALS_DIR, 
                                top_n = 20, tick_interval = 1):
    """
    Plot raw term‐count trends over a range of years for the top_n keywords 
    of a given cluster.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'Cluster', 'Abstract', and 'Year'
    cluster_id : int
        The cluster label to filter on
    all_kw : dict
        Mapping from cluster_id to {keyword: score, ...}
    cluster_name : str
        Human‐readable name used in the plot title and filename
    lower_yr : int
        Start year (inclusive)
    upper_yr : int
        End year (inclusive)
    temporal_palette : list
        A list of colors (length >= top_n) for plotting each keyword.
    VISUALS_DIR : str
        Directory path where the plot PNG will be saved.
    top_n : int, default = 20
        Number of top keywords to plot.
    tick_interval : int, default = 1
        Spacing between x‐axis ticks in years.

    Returns
    -------
    None
    
    """
    # Filter for only the given cluster (cluster_id)
    cluster_df  = df[df["Cluster"] == cluster_id]
    # Get the top_n keywords for this cluster
    cluster_kw  = list(all_kw[cluster_id].keys())[:top_n]
    # Build a list of years to iterate over
    years       = list(range(lower_yr, upper_yr + 1))
    # Restrict data to the year range
    cluster_df  = cluster_df[cluster_df["Year"].between(lower_yr, upper_yr)]
    # Count occurrences of each keyword by year
    term_counts = {term: [] for term in cluster_kw}
    
    for yr in years:
        # Concatenate all abstracts for that year
        text = (
            cluster_df[cluster_df["Year"] == yr]
            ["Abstract"].str.cat(sep=" ").lower())
        
        for term in cluster_kw:
            term_counts[term].append(text.count(term))
    filtered_terms = [t for t, counts in term_counts.items() if any(counts)]
    term_counts = {t: term_counts[t] for t in filtered_terms}
    cluster_kw = filtered_terms
    # Plot each keyword’s time series
    plt.figure(figsize = (8, 5))
    for color, term in zip(temporal_palette, cluster_kw):
        plt.plot(
            years,
            term_counts[term],
            label     = term,
            color     = color,
            linewidth = 1)

    # Format axes and title
    plt.xlim(lower_yr, upper_yr)
    plt.xticks(np.arange(lower_yr, upper_yr + 1, tick_interval))
    plt.xlabel("Publication Year", fontsize = 12)
    plt.ylabel("Raw Term Count", fontsize = 12)
    plt.title(f"{cluster_name} Topic: Keyword Trends ({lower_yr}–{upper_yr})",
              fontsize = 14)

    # Add legend outside plot
    plt.legend(
        bbox_to_anchor = (1.02, 1),
        loc = "upper left",
        frameon = False,
        fontsize = "small")

    plt.tight_layout()

    # Save the figure
    os.makedirs(VISUALS_DIR, exist_ok = True)
    temporal_plot_path = (
        f"{VISUALS_DIR}/{cluster_name.replace(' ', '_')}_"
        f"Trends_{lower_yr}_{upper_yr}_TEST.png")
    plt.savefig(temporal_plot_path, dpi = 300, bbox_inches = "tight")
    plt.show()