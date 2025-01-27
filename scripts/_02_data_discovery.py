import pandas as pd
from collections import Counter

# Read data
df = pd.DataFrame(pd.read_parquet('data/reuters_data.parquet'))

# Analyze labels
all_labels = [cat for cats in df['categories'] for cat in cats]
label_counts = Counter(all_labels)

# Print statistics
print(f"Distinct labels: {len(label_counts)}") #90
print(f"Total label assignments: {len(all_labels)}")
print(f"Average labels per document: {len(all_labels)/len(df):.2f}")

# Distribution of top 10 most common labels
print("\nTop 10 most common labels:")
for label, count in label_counts.most_common(10):
   print(f"{label}: {count} ({count/len(df):.1%} of documents)")