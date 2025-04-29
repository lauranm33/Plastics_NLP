### USE WITH SUPERCOMPUTER - SBATCH FILE IN SBATCH FOLDER IN SCRATCH ###

import fastparquet
import argparse
import pandas as pd
from tqdm import tqdm
from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc
import warnings
warnings.filterwarnings("ignore")

# Configure API client
config = {
    "apikey": "YOUR_API_KEY",
    "insttoken": "YOUR_INST_TOKEN"
}
client = ElsClient(config["apikey"])
client.inst_token = config["insttoken"]

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--start", type = int, required = True, help = "Start index for DOI extraction")
parser.add_argument("--end", type = int, required = True, help = "End index for DOI extraction")
args = parser.parse_args()

# Load the dataset
all_queries_path = "D:/MS_Thesis/API_Queries/All_Queries.gzip"
df = pd.read_parquet(all_queries_path)

# Create DOI-to-category mapping
doi_to_category = df.set_index("DOI")["Plastic_Category"].to_dict()

# Lists to store extracted data
list_abstract = []
list_doi = []
list_title = []
list_date = []
list_journal = []
list_category = []

doi_series = df["DOI"].reset_index(drop = True)  # Sequential index
max_index = len(doi_series) - 1  # Maximum valid index

# Adjust end index to not exceed available data
end = min(args.end, max_index + 1)
if args.start > max_index:
    print(f"Start index {args.start} exceeds available data (max index {max_index}). Exiting.")
    exit()
    

for i in tqdm(range(args.start, end)):
    try:
        current_doi = doi_series[i]
        doi_doc = FullDoc(doi=current_doi)
        if doi_doc.read(client):
            abstract = doi_doc._data["coredata"]["dc:description"]
            title = doi_doc.title
            date = doi_doc._data["coredata"]["prism:coverDisplayDate"]
            journal = doi_doc._data["coredata"]["prism:publicationName"]
            
            list_abstract.append(abstract)
            list_doi.append(current_doi)
            list_title.append(title)
            list_date.append(date)
            list_journal.append(journal)

        else:
            print(f"Operation failed for DOI: {current_doi}")
            # Append placeholders
            list_abstract.append("0")
            list_doi.append(current_doi)
            list_title.append("2")
            list_date.append("3")
            list_journal.append("4")

    except Exception as e:
        # Handle cases where i might be invalid (though adjusted)
        print(f"Error at index {i}: {e}")
        list_abstract.append("0")
        list_doi.append(doi_series[i] if i <= max_index else "Invalid DOI")
        list_title.append("2")
        list_date.append("3")
        list_journal.append("4")

# Create and save the extracted data
df_abstracts = pd.DataFrame({
    "DOI": list_doi,
    "Title": list_title,
    "Abstract": list_abstract,
    "Date": list_date,
    "Journal": list_journal,
})

output_path = f"D:/MS_Thesis/API_Queries/Abstracts/Abstracts_{args.start}_{end-1}.gzip"
df_abstracts.to_parquet(output_path, compression = "gzip", engine = "fastparquet")

print(f"Saved extracted abstracts from {args.start} to {end-1} in {output_path}")
