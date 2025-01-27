import nltk
import pandas as pd
from nltk.corpus import reuters

nltk.download('reuters')
nltk.download('punkt')

# Get documents and categories
documents = []
for fileid in reuters.fileids():
    text = reuters.raw(fileid)
    categories = reuters.categories(fileid)
    documents.append({
        'id': fileid,
        'text': text,
        'categories': categories
    })

# Create DataFrame and save to parquet
df = pd.DataFrame(documents)
df.to_parquet('data/reuters_data.parquet')

# create subset of the data for testing
# Subset to approximately 1/100 of the original size
df_subset = df.sample(frac=0.01, random_state=42)

# Save the subset to a new parquet file
df_subset.to_parquet('data/reuters_data_test.parquet')