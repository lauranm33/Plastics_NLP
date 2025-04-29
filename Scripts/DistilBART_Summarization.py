import torch
import nltk
import os, re, time
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from nltk.tokenize import sent_tokenize
from keybert import KeyBERT
from sklearn.cluster import KMeans
nltk.download(["stopwords", "punkt"], quiet = True)
import sys
SCRIPTS_DIR = r"D:\MS_Thesis\Scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from NLP_utils import fix_embedding_format
# =============================================================================
# Configuration
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
# Paths
INPUT_DATA = "D:/MS_Thesis/Models/Merged_Top100_With_Metadata.csv"
OUTPUT_DIR = "D:/MS_Thesis/DistilBART/Topic Summaries Preappended Topic Label/"
    
# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-6-6"  
    
# Processing
MIN_LENGTH = 150
MAX_LENGTH = 1024
NUM_FIXED_CHUNKS = 10
NUM_LABELS = 5
STOP_WORDS = set(stopwords.words("english")).union({"study", "research", "paper"})

# Manual Topic Labels
USER_TOPIC_LABELS = {
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
    
# =============================================================================
# Debugging utilities
# =============================================================================

DEBUG = True
def debug_print(message):
    if DEBUG:
        print(f"DEBUG {message}")
        
# =============================================================================
# Text Cleaning
# =============================================================================

def clean_text(text):
    """
    Clean up a raw text string by normalizing whitespace, fixing repeated
    periods, & removing non-ASCII characters.

    Parameters
    ----------
    text : str
        Raw input string to be cleaned.

    Returns
    -------
    str
        The cleaned text with:
            - Consecutive whitepace into a single space 
            - Runs of multiple periods replaced by a single period + space 
            - Non-ASCII characters removed
            - Leading/trailing whitespace stripped

    """
    
    # Collapse any sequence of whhitespace (tabs, newlines, multiple spaces)
    # into 1 space
    text = re.sub(r"\s+", " ", text)  
    # Replace sequences of periods (e.g. "...", ". . . ") w/ a period 
    # + space
    text = re.sub(r"(\.\s*)+", ". ", text)  # fix dot runs
    # Remove any characters outside basic ASCII range
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Strip leading & trailing whitespace & return
    return text.strip()

def clean_final_summary(summary):
    """
    Clean & normalize a generated summary string.

    Parameters
    ----------
    summary : str
        The raw summary text to be cleaned.

    Returns
    -------
    summary : str
        The cleaned summary with:
            - Non-ASCII characters removed,
            - Excess whitespace collapsed,
            - Consecutive duplicate words removed
            - Guaranteed ending punctuation (., !, or ?).

    """
    # Remove non-ASCII characters and normalize whitespace
    summary = re.sub(r"[^\x00-\x7F]+", " ", summary)
    summary = re.sub(r"\s+", " ", summary).strip()

    # Remove duplicate words (ignoring case & punctuation, e.g. commas)
    summary = re.sub(r"\b(\w+)([\s,]+(\1\b))+", r"\1", 
                     summary, flags = re.IGNORECASE)

    # Ensure the summary ends with a punctuation mark
    if not summary.endswith((".", "!", "?")):
        summary += "."

    return summary
# =============================================================================
# Extra Refinement Function for Final Summaries
# =============================================================================
def iterative_final_summary(summary, tokenizer, model, embedder, 
                            iterations = 3, similarity_threshold = 0.85):
    """
    Iteratively refine a generated summary by removing redundant statments &
    ensuring clarity.

    Parameters
    ----------
    summary : str
        The initial raw summary text to be refined.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the summarization model.
    model : transformers.PreTrainedModel
        Text generation model (e.g., DistilBART model) used in 'summarize_text'.
    embedder : sentence_transformers.SentenceTransformer
        Embedding model used to ID semantic similarity for duplicate removal.
    iterations : int, default = 3
        Number of refinement passes to apply. The default is 3.
    similarity_threshold : float, default = 0.85
        Cosine-similarity cutoff above which 2 sentences are duplicates
        & redundant.

    Returns
    -------
    refined : str
        The final cleaned de-duplicated summary after specific number of
        iterative refinements.

    """
    # Start from original summary
    refined = summary
    for i in range(iterations):
        # Prompt model to remove redundant phrases & clarify points.
        prompt = (
            "Refine the following summary by removing any redundant, repetitive, or duplicated phrases. "
            "Present each key point clearly and only once:\n\n" + refined)
        # Generate new summary based on the prompt
        refined = summarize_text(prompt, tokenizer, model)
        # Clean up whitespace, punctuation, & duplicate words
        refined = clean_final_summary(refined)
        # Further remove semantically similar sentences above defined threshold
        refined = refine_summary(refined, embedder, 
                                 similarity_threshold = similarity_threshold)
    # Return fully refined summary
    return refined

# =============================================================================
# Refine summary - redundancy & short sentence removal
# =============================================================================

def refine_summary(summary, embedder, similarity_threshold = 0.9):
    """
    Further refine summary by removing redundant or very short sentences.

    Parameters
    ----------
    summary : str
        The input summary text to be cleaned and deduplicated.
    embedder : sentence_transformers.SentenceTransformer
        Model used to encode each sentence into an embedding for 
        similarity checks.
    similarity_threshold : float, default = 0.9
        Cosine similarity above which two sentences are considered redundant.

    Returns
    -------
    str
        The refined summary text, formed by joining the filtered sentences.

    """

    # Split summary into individual sentences
    sentences = sent_tokenize(summary)
    if not sentences:
        return summary
    
    # Lists to hold final sentences & embeddings
    refined = []
    refined_embeds = []
    
    # Encode all sentences simultaneously (returns NumPy array of embeddings)
    sent_embeds = embedder.encode(sentences, convert_to_numpy = True)
    # Iterate over each sentence & its embedding
    for i, sentence in enumerate(sentences):
        # Skip short sentences
        if len(sentence.split()) < 5:
            continue
        
        add = True
        # Compare sentence embedding to each already kept embedding
        for j in range(len(refined)):
            sim = cosine_similarity([sent_embeds[i]], 
                                    [refined_embeds[j]])[0, 0]
            # If similarity exceeds threshold, consider it redundant
            if sim > similarity_threshold:
                add = False
                break
        # If not flagged as redundant, keep it
        if add:
            refined.append(sentence)
            refined_embeds.append(sent_embeds[i])
    # Reassumble the retained sentences into the final summary
    return " ".join(refined)

# =============================================================================
# Core functions
# =============================================================================

def load_models():
    """
    Load the summarization tokenizer, seq2seq model, and SBERT embedder.

    Returns
    -------
    tokenizer : transformers.AutoTokenizer
        Tokenizer loaded from the pre-trained summarization model identifier.
    model : transformers.AutoModelForSeq2SeqLM
        Sequence-to-sequence language model loaded and moved to the 
        specified DEVICE.
    embedder : sentence_transformers.SentenceTransformer
        SBERT embedder for semantic similarity tasks, configured to 
        run on DEVICE.

    """
    # Print to console to indicate that model loading has started
    debug_print("Loading models...")
    # Load tokenizer for summarization model
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
    # Load the seq2seq model and move it to the desired device (CPU/GPU)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL).to(DEVICE)
    # Initialize SBERT embedder on same device
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device = DEVICE)
    return tokenizer, model, embedder

# =============================================================================
# KeyBERT Subtheme Labeling Function
# =============================================================================

def label_subtheme(summary, kw_model, num_labels = NUM_LABELS):
    """
    Extract a single concise subtheme label from a summary using KeyBERT.

    Parameters
    ----------
    summary : str
        The text from which to derive a subtheme label.
    kw_model : KeyBERT
        A KeyBERT instance configured for keyword extraction.
    num_labels : int, default = NUM_LABELS
        Number of candidate keywords to consider; the top one is chosen. 
        The default is NUM_LABELS.

    Returns
    -------
    str
        The top keyword phrase (title-cased) if any candidates are found,
        otherwise "Unlabeled Subtheme".

    """
    
    # Extract candidate phrases (1-3 words) with diversity constraint
    candidates = kw_model.extract_keywords(
        summary,
        keyphrase_ngram_range = (1, 3),
        stop_words = list(STOP_WORDS),
        top_n = NUM_LABELS,
        diversity = 0.7)
    # If any candidates are extracted, return highest-ranked one, title-cased
    if candidates:
        return candidates[0][0].title()
    # Otherwise, fall back to default placeholder
    else:
        return "Unlabeled Subtheme"

# =============================================================================
# Generate overall summary labels using KeyBERT
# =============================================================================

def generate_final_topic_labels(overall_summary, 
                                kw_model, 
                                num_labels = NUM_LABELS):
    
    """
    Extract multiple candidate labels for an overall topic summary 
    using KeyBERT.

    Parameters
    ----------
    overall_summary : str
       The aggregated summary text for which to generate labels.
    kw_model : KeyBERT
        A KeyBERT instance configured for keyword extraction.
    num_labels : int, default = NUM_LABELS
        Number of top candidate labels to return.

    Returns
    -------
    list of str
        A list of title-cased label strings corresponding to the 
        top candidate phrases.
    
    """

    candidates = kw_model.extract_keywords(
        overall_summary,
        keyphrase_ngram_range = (1, 3),
        stop_words = list(STOP_WORDS),
        top_n = NUM_LABELS,
        diversity = 0.7)
    
    # Title-case each phrase and return the list
    labels = [phrase.title() for phrase, score in candidates]
    return labels

# =============================================================================
# Summarization functions
# =============================================================================

def summarize_text(text, tokenizer, model):
    """
    Generate a summary for given text using a seq2seq model.

    Parameters
    ----------
    text : str
        Input text to be summarized.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer corresponding to the seq2seq model.
    model : transformers.PreTrainedModel
        The loaded sequence-to-sequence model for summarization.

    Returns
    -------
    str
        The decoded summary string, with special tokens removed.

    """
    
    # Tokenize and convert to  tensors or pad to MAX_LENGTH
    inputs = tokenizer(text, 
                       max_length = MAX_LENGTH, 
                       truncation = True, 
                       return_tensors = "pt").to(DEVICE)
    # Generate summary tokens
    outputs = model.generate(
        inputs['input_ids'], 
        max_length = MAX_LENGTH,
        min_length = MIN_LENGTH,
        no_repeat_ngram_size = 3,
        early_stopping = True)
    # Decode 1st generated sequence into text
    return tokenizer.decode(outputs[0], skip_special_tokens = True)

def get_summarizer():
    """
    Initiate a HuggingFace summarization pipeline on the available device.

    Returns
    -------
    transformers.Pipeline
        A ready-to-use summarization pipeline using SUMMARIZATION_MODEL.
        
    """
    
    # Use GPU device 0 if available, else -1 for CPU.
    device_num = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", 
                    model = SUMMARIZATION_MODEL, 
                    device = device_num)

# =============================================================================
# Chunking functions
# =============================================================================

def summarize_chunk(chunk_text, chunk_id, prefix_label):
    """
    Summarize a single text chunk with a subtheme prompt
    with error handling.
    
    Parameters
    ----------
    chunk_text : str
        The raw text of the chunk to summarize.
    chunk_id : int
        An identifier for this chunk, used in error messages.
    prefix_label : str
        A label or prefix indicating the topic (manually defined by user from
        BERTopic Results) for context in the prompt.

    Returns
    -------
    str
        The generated summary (or the first 500 characters of the 
                               input on failure).
    
    """
    
    summarizer = get_summarizer()
    # Prepend subtheme label to chunk text for model context
    chunk_input = f"{prefix_label}: {chunk_text}"
    
    try:
        result = summarizer(
            chunk_input,
            max_length = MAX_LENGTH,
            min_length = MIN_LENGTH,
            do_sample = False,
            truncation = True,
            no_repeat_ngram_size = 3,
            early_stopping = True)
        
        # Handle both cases where result might be a list or dict
        if isinstance(result, list) and len(result) > 0:
            summary = result[0].get("summary_text", chunk_text[:500])
        else:
            summary = result.get("summary_text", chunk_text[:500])
        return summary
    except Exception as e:
        print(f"Chunk {chunk_id} summarization failed: {e}")
        # Fallback: return first 500 characters of raw chunk.
        return chunk_text[:500]
    
# =============================================================================
# Post-process generated summaries 
# =============================================================================
def postprocess_summary(summary):
    """
    Post-process a generated summary by normalizing whitespace & punctuation,
    removing invalid tokens, stripping prompt remnants, & apply final cleanup.

    Parameters
    ----------
    summary : str
        The raw summary text to be post-processed.

    Returns
    -------
    str
        The cleaned and finalized summary.
    """
    # Remove extra whitespace and duplicate punctuation
    summary = re.sub(r"\s+", " ", summary)
    summary = re.sub(r"(\.\s*){2,}", ". ", summary)
    # Remove quoted standalone numbers (e.g., "'3'")
    summary = re.sub(r"'\d+'", "", summary)
    # Remove any remnants of prompt text if present (case-insensitive)
    summary = re.sub(r"(Avoid repetition and fragmented sentences\.?\s*)", "", 
                     summary, flags = re.IGNORECASE)
    # Apply final cleaning (ensures proper punctuation and whitespace)
    return clean_final_summary(summary)

# =============================================================================
# Parallel summarization
# =============================================================================
def summarizer_worker(args):
    """
    Summarize a single text chunk in parallel, tagging it with its chunk ID
    and subtheme label and return cleaned summary or a failure token.
    
    Parameters
    ----------
    args : tuple
        A 4-element tuple containing:
        - chunk_text (str): The raw text of the chunk to summarize.
        - chunk_id (int): Identifier for this chunk (used in logging).
        - topic_label (str): Main topic label (manually defined by
          user based on BERTopic results) to prepend in the prompt.
        - subtheme_label (str): The subtheme label for metadata mapping.

    Returns
    -------
    tuple
        (chunk_id, subtheme_label, summary_text) where:
        - chunk_id (int): The same ID passed in.
        - subtheme_label (str): The same subtheme label passed in used 
        for metadata labeling.
        - summary_text (str): The post-processed summary, or
        "[SUMMARY FAILED]" on error.
   
    """
    
    chunk_text, chunk_id, topic_label, subtheme_label = args
    try:
        # Generate raw summary for current chunk using topic_label as prefix
        summary = summarize_chunk(chunk_text, chunk_id, topic_label)
        # Handle case where summarization returns None
        if summary is None:  
            summary = "[SUMMARY FAILED]"
        # Post process and return chunk ID, subtheme label for metadata & 
        # clean summary
        return chunk_id, subtheme_label, postprocess_summary(summary)
    
    except Exception as e:
        # Log error & return failture token
        print(f"Chunk {chunk_id} summarization failed: {e}")
        return chunk_id, subtheme_label, "[SUMMARY FAILED]"
    
def create_grouped_chunks(documents, embedder, num_groups = NUM_FIXED_CHUNKS):
    """
    Cluster a list of documents into a fixed number of groups and concatenate 
    each group's texts into a single chunk.

    Parameters
    ----------
    documents : list of str
        The pre-tokenized document texts to group.
    embedder : sentence_transformers.SentenceTransformer
        Model used to encode each document into an embedding.
    num_groups : int, default = NUM_FIXED_CHUNKS
        Desired number of clusters/chunks. If greater than the number of 
        documents, it is reduced to len(documents).

    Returns
    -------
    list of str
        A list of concatenated chunk strings, one per cluster, 
        in ascending label order.
    
    """
    # If no documents, then nothing to group.
    if not documents:
        return []
    # Encode all docs into embeddings for clustering
    doc_embeds = embedder.encode(documents, convert_to_numpy = True)
    # Determine number of clusters (cannot exceed number of docs)
    n = len(documents)
    k = min(num_groups, n)
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters = k, random_state = 42)
    labels = kmeans.fit_predict(doc_embeds)
    # Group docs by their assigned label
    groups = {}
    for label, doc in zip(labels, documents):
        groups.setdefault(label, []).append(doc)
    # Concatenate each group's docs into 1 chunk (= 1 subtheme)
    chunks = []
    for label in sorted(groups.keys()):
        chunk_text = " ".join(groups[label])
        chunks.append(chunk_text)
    return chunks

# =============================================================================
# Subtheme Summarization: Grouped Chunking with Parallel Processing
# =============================================================================
def summarize_topic_with_subthemes_grouped(
        filtered_docs, tokenizer, summarization_model, topic_id, kw_model, 
        embedder, num_groups = NUM_FIXED_CHUNKS):
    """
    Summarize a topic by clustering its documents into subthemes, 
    summarize each subtheme, label with KeyBERT, & then generate
    an overall theme summary.

    Parameters
    ----------
    filtered_docs : list of str
        Preprocessed documents belonging to the topic.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the sequence-to-sequence summarization model.
    summarization_model : transformers.PreTrainedModel
        The seq2seq model used to generate summaries.
    topic_id : int
        Identifier of the topic being processed.
    kw_model : KeyBERT
        KeyBERT instance for extracting subtheme labels.
    embedder : sentence_transformers.SentenceTransformer
        Embedding model for grouping documents into clusters.
    num_groups : int, default = NUM_FIXED_CHUNKS
        Number of subtheme clusters to create.

    Returns
    -------
    overall_summary : str
        The final summary for the topic.
    subtheme_summaries : list of str
        Individual summarized entries for each subtheme, prefixed by manually
        defined user labels (using BERTopic results)
    
    """
    
    # Use manually defined user labels for current topic
    topic_label = USER_TOPIC_LABELS.get(topic_id, f"Topic_{topic_id}")
    
    # Cluster the documents into 'num_groups' subtheme chunks
    chunks = create_grouped_chunks(filtered_docs, 
                                   embedder, 
                                   num_groups = num_groups)
    debug_print(f"Grouped chunking: created {len(chunks)} chunks for subtheme summarization.")
    # Generate a subtheme label for each chunk    
    subtheme_labels = [label_subtheme(chunk, kw_model) for chunk in chunks]
    subtheme_summaries = []
    for i, chunk in enumerate(chunks):
        # Generate subtheme summary
        _, _, result = summarizer_worker((chunk, 
                                          i, 
                                          topic_label, 
                                          subtheme_labels[i]))
        # Build a human-readable summary entry for this subtheme
        subtheme_summary_entry = f"{topic_label} - Subtheme {i+1}: {subtheme_labels[i]} - {result}"
        subtheme_summaries.append(subtheme_summary_entry)
        
        # Append this subtheme summary to a CSV for later inspection
        pd.DataFrame([{
            "Topic_ID": topic_id,
            "Subtheme_ID": i+1,
            "Subtheme_Label": subtheme_labels[i],
            "Subtheme_Summary": result
            }]).to_csv(
            os.path.join(OUTPUT_DIR, f"Topic_{topic_id}_Subthemes.csv"),
            mode = "a",  # Append mode
            header = not os.path.exists(
                os.path.join(OUTPUT_DIR, 
                             f"Topic_{topic_id}_Subthemes.csv")),  
            index = False)
    # Combine all subtheme entries into 1 block for overall summary
    combined_subtheme_text = "\n".join(subtheme_summaries)
    debug_print("Generating overall theme summary from subtheme summaries...")
    # Generate overall topic summary from combined subthemes
    overall_summary = summarize_text(
        combined_subtheme_text, 
        tokenizer, 
        summarization_model)
    # Clean up and normalize overall summary
    overall_summary = clean_final_summary(overall_summary)
    # Refine the summary through iterative passes to remove redundancy
    overall_summary = iterative_final_summary(
        overall_summary, 
        tokenizer, 
        summarization_model, 
        embedder)
                
    return overall_summary, subtheme_summaries

# =============================================================================
# Process one topic (subtheme summaries & final topic labels)
# =============================================================================

def process_topic(topic_df, tokenizer, summarization_model, embedder, kw_model):
    """
    Process a single topic's DataFrame to filter documents, generate subtheme
    sumaries, & produce final topic labels. 

    Parameters
    ----------
    topic_df : pandas.DataFrame
        DataFrame containing rows for one topic, with columns:
        - "Topic": integer topic ID (same for all rows in 1 cluster)
        - "Representative_Abstract": text abstracts for summarization
        - "Embedding": precomputed embedding (either str or array-like)
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the summarization model.
    summarization_model : transformers.PreTrainedModel
        Sequence-to-sequence model for generating summaries.
    embedder : sentence_transformers.SentenceTransformer
        Embedding model used for clustering and filtering.
    kw_model : KeyBERT
        Model used to extract subtheme and topic labels.
        
    Returns
    -------
    overall_summary : str or None
        The consolidated summary of the topic after subtheme processing;
        None if an error occurred.
    topic_label : str or None
        Human-readable label for the topic from USER_TOPIC_LABELS;
        None if an error occurred.
    subtheme_summaries : list of str or None
        Summaries for each detected subtheme;
        None if an error occurred.
    final_topic_labels : list of str or None
        Final candidate labels extracted from the overall summary;
        None if an error occurred.

    """

    try:
        # Extract the numeric topic ID
        topic_id = int(topic_df["Topic"].iloc[0])
        # Extract topic's human-readable label
        topic_label = USER_TOPIC_LABELS[topic_id]
        # Extract representative abstracts for this topic
        documents = topic_df["Representative_Abstract"].tolist()
        # Reconstruct embeddings: parse strings or use arrays directly
        embedding_list = []
        for emb in topic_df["Embedding"].tolist():
            if isinstance(emb, str):
                # Convert str to NumPy array
                embedding_list.append(fix_embedding_format(emb))
            else:
                # Already an array-like embedding
                embedding_list.append(np.array(emb))
        embeddings = np.stack(embedding_list)
        
        # Filter out low-similarity docs w/ cosine similarity to centroid
        debug_print("Filtering documents using precomputed embeddings...")
        centroid = np.mean(embeddings, axis = 0)
        sim_scores = cosine_similarity(embeddings, centroid.reshape(1, -1))
        filtered_idx = np.where(sim_scores > 0.5)[0]
        filtered_docs = [documents[i] for i in filtered_idx]
        debug_print(f"Selected {len(filtered_docs)}/{len(documents)} documents.")

        # Summarize the filtered documents by subthemes and then overall
        overall_summary, subtheme_summaries = summarize_topic_with_subthemes_grouped(
            filtered_docs, tokenizer, summarization_model, 
            topic_label, kw_model, embedder)
        
        # Generate final candidate labels for overall summary
        final_topic_labels = generate_final_topic_labels(
            overall_summary, kw_model, num_labels = NUM_LABELS)
        
        # Return all results for this topic cluster
        return overall_summary, topic_label, subtheme_summaries, final_topic_labels

    except Exception as e:
        # Log any error and return None placeholders
        print(f"ERROR processing topic: {str(e)}")
        return None, None, None, None

# =============================================================================
# Main Execution
# =============================================================================

def process_all_topics():
    """
    1. Load dataset of documents (abstracts) & corresponding SBERT embeddings.
    2. Normalize embedding strings into NumPy arrays (NLP_utils.py).
    3. Initialize models (summarizer & embedder) and KeyBERT.
    4. Iterate over each topic group:
        i. Filter & summarize with process_topic.
        ii. Print summaries & labels to console.
        iii. Write each topic's results to a .csv file.
        
    No parameters. Relies on global constants:
        - INPUT_DATA : path to CSV with 'Topic', 'Embedding', and
                      'Representative Abstract' columns.
        - OUTPUT_DIR : directory to save per-topic summary CSVs.
        - NUM_FIXED_CHUNKS : Default number of subtheme clusters to create.
        - NUM_LABELS : Number of candidate labels to extract per topic using
                       KeyBERT.
        - USER_TOPIC_LABELS : Dict mapping topic IDs to user defined labels.
        - Other pipeline settings (e.g. MAX_LENGTH, MIN_LENGTH) pulled from
          globals.

    Returns
    -------
    None.

    """
    # 1. Load full input DataFrame
    df = pd.read_csv(INPUT_DATA)
    # 2. Convert str embeddings to NumPy arrays.
    df["Embedding"] = df["Embedding"].apply(fix_embedding_format)
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 3. Load tokenizer, seq2seq summarization model, SBERT embedder & KeyBERT
    tokenizer, summarization_model, embedder = load_models()
    # KeyBERT for subtheme labeling
    kw_model = KeyBERT(model = embedder)
    # Group DataFrame by "Topic" column
    topic_groups = df.groupby("Topic")
    # 4. Iterate over each topic group
    for topic, group in topic_groups:
        print(f"\nProcessing Topic {topic} with {len(group)} documents:")
        # i. Filter & summarize with process_topic (per-topic pipeline)
        overall_summary, topic_label, subtheme_summaries, final_topic_labels = process_topic(group, tokenizer, summarization_model, embedder, kw_model)
        # ii. Print summaries & labels to console.
        print(f"Topic Label: {topic_label}")
        print("Overall Summary:")
        print(overall_summary)
        print("Subtheme Summaries:")
        for s in subtheme_summaries:
            print(s)
        print("Final Candidate Topic Labels for Overall Theme:")
        print(", ".join(final_topic_labels))
        print("\n" + "-"*80 + "\n")
        # iii. Write each topic's results to a .csv file.
        pd.DataFrame([{
            "Topic_ID": topic,
            "Topic_Labels": topic_label,
            "Final_Topic_Labels": ", ".join(final_topic_labels),
            "Overall_Summary": overall_summary,
            "Subtheme_Summaries": "\n".join(subtheme_summaries),
            "Documents_Used": len(group)
        }]).to_csv(os.path.join(OUTPUT_DIR, f"Topic_{topic}_Summary.csv"), 
                   index = False)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    process_all_topics()
#%% Post-processing topics
# Loop over topic IDs
topic_id = np.arange(0, 14, 1)
for i in topic_id:
    # Read previously saved CSV file
    df = pd.read_csv(f"{OUTPUT_DIR}/Topic_{i}_Summary.csv")
    topic_label = USER_TOPIC_LABELS.get(i, f"Topic {i}")
    # Extract & split big Subtheme_Summaries string into individual 
    # subtheme entries
    subtheme_text = df.at[0, "Subtheme_Summaries"]
    # Split into individual subthemes
    subthemes = re.split(r"Subtheme \d+: ", subtheme_text)[1:]
    # Initialize lists for labels and summaries
    topic_labels = []
    labels = []
    summaries = []
    
    for subtheme in subthemes:
        # Handle quoted labels
        if subtheme.startswith("'"):
            # Split into label and summary using first pair of quotes
            parts = re.match(r"'([^']+)'(.+)", subtheme, re.DOTALL)
            if parts:
                label = parts.group(1).strip().rstrip('-').strip()
                summary = parts.group(2).strip()
            else:
                label = ""
                summary = subtheme.strip()
        else:
            # Handle unquoted labels with dash separation
            parts = re.split(r' - ', subtheme, 1)
            if len(parts) > 1:
                label = parts[0].strip()
                summary = parts[1].strip()
            else:
                label = ""
                summary = subtheme.strip()

        topic_labels.append(topic_label)
        labels.append(label)
        summaries.append(summary)
    
    result_df = pd.DataFrame({
        'Cluster Topic Label': topic_labels,
        'Subtheme Label': labels,
        'Subtheme Summary': summaries})

    # Save to a new CSV file
    result_df.to_csv(f"{OUTPUT_DIR}/Topic_{i}_Subthemes.csv", index = False)
    
    print("Subthemes extracted and saved to Topic_{i}_Subthemes.csv")