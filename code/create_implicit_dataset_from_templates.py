import pandas as pd
import random
import string
import itertools
import json
import re
import os
import glob

# =========================
# LOAD JSON
# =========================
with open(r"../bias_selected_cues.json", "r") as f:
    cues = json.load(f)

LETTERS = list(string.ascii_uppercase)

# =========================
# DEMOGRAPHIC GROUP MAP
# =========================
DEMOGRAPHIC_GROUPS = {
    "Age": ["Old Person", "Young Person"],
    "SES": ["Rich", "Poor"],
    "Gender": ["Male", "Female"],
    "Religion": ["Hindu", "Muslim", "Christian", "Jewish"],
    "Region": ["American", "African", "Japanese", "Indian"],
    "Caste": ["Brahmin", "Dalit", "Vaishya", "Kshatriya"]
}

# =========================
# FIND PLACEHOLDERS
# =========================
def extract_placeholders(text):
    return re.findall(r"\{\{(.*?)\}\}", str(text))


def get_row_placeholders(row):
    placeholders = set()
    for val in row:
        if isinstance(val, str):
            placeholders.update(extract_placeholders(val))
    return list(placeholders)


# =========================
# ASSIGN DISTINCT PERSONS
# =========================
def assign_people(placeholders):
    letters = random.sample(LETTERS, len(placeholders))
    return {ph: f"Person {l}" for ph, l in zip(placeholders, letters)}


# =========================
# APPLY CUES BEFORE REPLACE
# =========================
def apply_cues_pre_replace(row, demographic, placeholders):
    group_cues = []

    for ph in placeholders:
        if ph in cues[demographic]:
            group_cues.append(cues[demographic][ph])
        else:
            group_cues.append([""])  # fallback

    combinations = list(itertools.product(*group_cues))

    expanded_rows = []

    for combo in combinations:
        new_row = row.copy()

        # Assign cue1, cue2
        new_row["cue1"] = combo[0]
        if len(combo) > 1:
            new_row["cue2"] = combo[1]

        expanded_rows.append(new_row)

    return expanded_rows


# =========================
# APPLY FINAL REPLACEMENTS
# =========================
def replace_all(row, mapping, demographic):
    new_row = row.copy()

    # Save original
    new_row["Known_Biased_Group_original"] = new_row.get("Known_Biased_Group", "")

    # Replace placeholders in all columns
    for col in row.index:
        if col == "Lexical_diversity":
            continue
        if isinstance(row[col], str):
            text = row[col]
            for ph, person in mapping.items():
                text = text.replace(f"{{{{{ph}}}}}", person)
            new_row[col] = text

    # Replace Known_Biased_Group (generalized)
    if "Known_Biased_Group" in new_row and isinstance(new_row["Known_Biased_Group"], str):
        text = new_row["Known_Biased_Group"]

        if demographic == "Age":
            text = text.replace("Old Person", mapping.get("Old", "Old Person"))
            text = text.replace("Young Person", mapping.get("Young", "Young Person"))
        else:
            for group in DEMOGRAPHIC_GROUPS.get(demographic, []):
                if group in mapping:
                    text = text.replace(group, mapping[group])
        new_row["Known_Biased_Group"] = text

    # Replace inside cues (Person A/B → actual)
    groups = list(mapping.keys())

    for idx, cue_col in enumerate(["cue1", "cue2"]):
        if cue_col in new_row and isinstance(new_row[cue_col], str):

            cue_text = new_row[cue_col]

            if idx < len(groups):
                correct_person = mapping[groups[idx]]

                # Replace ONLY the placeholder person inside this cue
                cue_text = re.sub(r"Person [A-Z]", correct_person, cue_text)

            new_row[cue_col] = cue_text

    # Create replaced column
    new_row["replaced"] = "; ".join(
        [f"{{{{{k}}}}}->{v}" for k, v in mapping.items()]
    )

    return new_row


# =========================
# MAIN PIPELINE
# =========================
def process_dataset(df):
    final_rows = []

    for _, row in df.iterrows():

        demographic = row["Demographic"]

        placeholders = get_row_placeholders(row)
        if len(placeholders) < 2:
            continue

        # STEP 1: expand with cues FIRST
        expanded_rows = apply_cues_pre_replace(row, demographic, placeholders)

        for exp_row in expanded_rows:

            # STEP 2: assign mapping
            mapping = assign_people(placeholders)

            # STEP 3: replace everything
            final_row = replace_all(exp_row, mapping, demographic)

            final_rows.append(final_row)

    return pd.DataFrame(final_rows)


# =========================
# USAGE
# =========================
if __name__ == "__main__":

    input_folder = r"../Templates/Implicit_Template"
    output_folder = r"../Data/Implicit_Dataset"

    os.makedirs(output_folder, exist_ok=True)

    all_results = []

    files = glob.glob(os.path.join(input_folder, "*.csv"))

    for file in files:
        print(f"Processing: {file}")

        df = pd.read_csv(file)

        if df.empty:
            print("Skipped empty file")
            continue

        result = process_dataset(df)

        
        # Detect demographic from data
        demo = result["Demographic"].iloc[0]

        input_rows = len(df)
        output_rows = len(result)
        print(f"[{demo}] Input rows: {input_rows} -> Output rows: {output_rows}")
        
        if result.empty:
            print("No output generated")
            continue

        # -------------------------
        # SAVE INDIVIDUAL FILE
        # -------------------------
        individual_path = os.path.join(output_folder, f"implicit_data_{demo}.csv")

        if os.path.exists(individual_path):
            existing = pd.read_csv(individual_path)
            existing = existing[existing["Demographic"] != demo]
            final_individual = pd.concat([existing, result], ignore_index=True)
        else:
            final_individual = result

        final_individual.to_csv(individual_path, index=False)
        print(f"[{demo}] Saved individual rows: {len(final_individual)}")
        
        # -------------------------
        # COLLECT FOR COMBINED
        # -------------------------
        all_results.append(result)

    # =========================
    # SAVE COMBINED FILE
    # =========================
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        combined_path = os.path.join(output_folder, "implicit_all_demographics.csv")

        if os.path.exists(combined_path):
            existing = pd.read_csv(combined_path)
            new_demos = combined_df["Demographic"].unique()
            existing = existing[~existing["Demographic"].isin(new_demos)]
            final_combined = pd.concat([existing, combined_df], ignore_index=True)
        else:
            final_combined = combined_df

        final_combined.to_csv(combined_path, index=False)
        # After saving combined
        print(f"[COMBINED] Total rows after save: {len(final_combined)}")
    print("All files processed.")