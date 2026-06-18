# -*- coding: utf-8 -*-
import pandas as pd
import json
import os

METADATA_FILE = "data/metadata.csv"

def analyze():
    if not os.path.exists(METADATA_FILE):
        print("Metadata file not found.")
        return

    df = pd.read_csv(METADATA_FILE, dtype=str).fillna("")
    
    # 1. Filter new data (unlabeled)
    new_df = df[df["review_status"] == "unlabeled"].copy()
    existing_df = df[df["review_status"] == "labeled"].copy()
    
    print(f"Total: {len(df)}")
    print(f"Existing (labeled): {len(existing_df)}")
    print(f"New (unlabeled): {len(new_df)}")

    # 2. Analyze new data distributions (estimated)
    # We used 'notes' field to store "Targeted: cat_name"
    new_df["est_cat"] = new_df["notes"].str.replace("Targeted: ", "")
    new_dist = new_df["est_cat"].value_counts().to_dict()
    
    # 3. Check priority categories
    priority_cats = ["equipment", "skill_tree", "shop", "crafting", "lobby", "pause_menu"]
    
    # 4. Total Distribution
    # Combine existing labeled counts with new estimated counts
    existing_dist = existing_df["primary_screen_type"].value_counts().to_dict()
    
    all_cats = set(list(existing_dist.keys()) + list(new_dist.keys()))
    total_dist = {}
    for cat in all_cats:
        if cat == "" or cat == "other": continue
        count = int(existing_dist.get(cat, 0)) + int(new_dist.get(cat, 0))
        total_dist[cat] = count

    # 5. Deficit list (< 20)
    deficit_list = {cat: count for cat, count in total_dist.items() if count < 20}

    # 6. Suspicious data (mismatch)
    # Simple check: if estimated category is not mentioned in caption at all
    suspicious = []
    for idx, row in new_df.iterrows():
        cat = row["est_cat"]
        caption = row["screenshot_caption"].lower()
        if cat != "other" and cat not in caption:
            # Check synonyms
            synonyms = {
                "skill_tree": ["ability", "skill", "talent", "perk", "upgrade"],
                "equipment": ["gear", "loadout", "weapon", "armor", "item"],
                "shop": ["store", "merchant", "buy", "sell", "vendor"],
                "crafting": ["forge", "recipe", "workshop"],
                "lobby": ["room", "multiplayer", "party"],
                "pause_menu": ["pause", "options", "quit"],
                "quest": ["mission", "objective", "journal"],
                "inventory": ["item", "backpack"],
                "loading_screen": ["loading"],
                "battle_result": ["result", "victory", "defeat", "reward"],
            }
            match = False
            for syn in synonyms.get(cat, [cat]):
                if syn in caption:
                    match = True
                    break
            if not match:
                suspicious.append(row)

    suspicious_df = pd.DataFrame(suspicious)
    suspicious_df.to_csv("targeted_collection_report.csv", index=False, encoding="utf-8-sig")
    
    # 7. Class Balance Report
    report_rows = []
    for cat in sorted(total_dist.keys()):
        report_rows.append({
            "category": cat,
            "existing_labeled": existing_dist.get(cat, 0),
            "new_collected": new_dist.get(cat, 0),
            "total_estimated": total_dist.get(cat, 0),
            "status": "OK" if total_dist.get(cat, 0) >= 20 else "DEFICIT"
        })
    
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv("class_balance_report.csv", index=False, encoding="utf-8-sig")

    # Print summary for the user
    print("\n--- Collection Summary ---")
    print(f"Total new items: {len(new_df)}")
    print("\n[Priority Category Boost]")
    for cat in priority_cats:
        print(f"  - {cat}: +{new_dist.get(cat, 0)} (Total Est: {total_dist.get(cat, 0)})")
    
    print("\n[Still Deficit (<20)]")
    for cat, count in deficit_list.items():
        print(f"  - {cat}: {count}")
    
    print(f"\nSuspicious items found: {len(suspicious_df)}")
    print("Reports saved: targeted_collection_report.csv, class_balance_report.csv")

if __name__ == "__main__":
    analyze()
