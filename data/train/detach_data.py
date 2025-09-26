import json
from collections import defaultdict

def process_data(data):
    """
    Processes a list of dictionaries to split the 'rejected' list into individual entries.
    Excludes beauty, sentiment, sycophancy, and gptinst bias types.
    Merges all remaining data (original + valid bias types) into a single dataset.
    Each entry gets a bias_type field indicating its specific category.

    Args:
        data: A list of dictionaries, where each dictionary contains 'question',
              'chosen', 'similar_question', 'rejected', and 'bias_type' keys. 
              The 'rejected' value is a list of rejected answers.

    Returns:
        A list of processed entries combining all valid categories.
    """
    # Bias types to exclude
    excluded_bias_types = {'beauty', 'sentiment', 'sycophancy'}
    
    all_entries = []
    
    for entry in data:
        base_entry = {
            "question": entry["question"],
            "chosen": entry["chosen"]
        }
        if "similar_question" in entry:
            base_entry["similar_question"] = entry["similar_question"]

        # Process each rejected answer with its corresponding category
        for i, rejected_answer in enumerate(entry["rejected"]):
            # Determine category based on index
            if i == 0:
                # Original rejected answer - always include
                category = "original"
                include_entry = True
            elif i == 1:
                # gptinst rejected answer - exclude
                category = "gptinst"
                include_entry = False
            elif i == 2:
                # First bias type
                if "bias_type" in entry and len(entry["bias_type"]) > 0:
                    category = entry["bias_type"][0]
                    include_entry = category not in excluded_bias_types
                else:
                    category = f"bias_type1"
                    include_entry = True
            elif i == 3:
                # Second bias type
                if "bias_type" in entry and len(entry["bias_type"]) > 1:
                    category = entry["bias_type"][1]
                    include_entry = category not in excluded_bias_types
                else:
                    category = f"bias_type2"
                    include_entry = True
            else:
                # For any additional rejected answers beyond index 3
                category = f"bias_type{i-1}"
                include_entry = True
            
            # Only include if not in excluded categories
            if include_entry:
                new_entry = base_entry.copy()
                new_entry["rejected"] = [rejected_answer]
                new_entry["bias_type"] = category  # Set the specific bias type for this entry
                all_entries.append(new_entry)
    
    return all_entries

# Read input data
with open("train_data.jsonl", "r", encoding="utf-8") as fin:
    lines = [json.loads(line.strip()) for line in fin.readlines()]
    processed_entries = process_data(lines)

# Write all entries to a single file
output_filename = "train_data_effective.jsonl"
with open(output_filename, "w", encoding="utf-8") as fout:
    for item in processed_entries:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Created {output_filename} with {len(processed_entries)} entries")

# Count entries by category for summary
category_counts = defaultdict(int)

for entry in processed_entries:
    category = entry["bias_type"]
    category_counts[category] += 1

print(f"\nSummary of included entries:")
print(f"Total entries: {len(processed_entries)}")
print(f"Breakdown by category:")
for category, count in sorted(category_counts.items()):
    print(f"  {category}: {count} entries")

excluded_bias_types = {'beauty', 'sentiment', 'sycophancy'}
print(f"\nExcluded bias types: {', '.join(excluded_bias_types)} and gptinst")
