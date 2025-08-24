import json
from collections import Counter

# File path
file_path = r"C:\Users\Kushal\Desktop\new\civil_cases.json"

# Load the JSON file
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Total number of cases
total_cases = len(data)

# Extract categories (sub-categories)
categories = [case.get("category", "Unknown") for case in data]

# Count categories
category_counts = Counter(categories)

# Print results
print(f"Total cases: {total_cases}\n")

print("Sub-category counts (Descending):")
for category, count in category_counts.most_common():  # <-- sorts by count (descending)
    print(f"{category}: {count}")
