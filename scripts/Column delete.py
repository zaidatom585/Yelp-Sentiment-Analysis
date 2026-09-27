import pandas as pd

input_path = "output/yelp_10k_sample_10000_20251216_003532.csv"  # raw sample from Importer_10k_reviews.py
output_path = "output/cleaned_yelp.xlsx"

df = pd.read_csv(input_path)

keep_columns = [
    'business_name', 'city', 'state', 'latitude', 'longitude', 'address',
    'business_rating', 'review_count_reviews', 'total_useful',
    'categories', 'price', 'sentiment', 'avg_review_stars'
]

df_cleaned = df[keep_columns]

df_cleaned.to_excel(output_path, index=False)

print(f"Cleaned file saved to {output_path}")
