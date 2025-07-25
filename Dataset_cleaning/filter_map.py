import pandas as pd

# File paths
input_csv = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/dmAP_all.csv"
output_csv = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/map_filtered_.7.csv"

# Load the CSV file
df = pd.read_csv(input_csv)
print("Available columns in CSV:", df.columns)

# Filter out rows where mAP_0.8 is between 0.3 and 0.8
df_filtered = df[(df["mAP_0.7"] > 0.3) & (df["mAP_0.7"] < 0.8)]

# Save the filtered data to a new CSV file
df_filtered.to_csv(output_csv, index=False)

print(f"Filtered data saved to {output_csv}")
