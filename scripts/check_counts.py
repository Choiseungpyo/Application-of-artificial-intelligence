import csv
from collections import Counter

TARGET_CLASSES = [
    "skill_tree", "lobby", "equipment", "pause_menu", 
    "crafting", "shop", "quest", "loading_screen", "battle_result"
]

counts = Counter()
with open('data/metadata.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        primary = row.get('primary_screen_type', '')
        target = row.get('source_target_screen_type', '')
        
        if primary in TARGET_CLASSES:
            counts[primary] += 1
        elif target in TARGET_CLASSES:
            counts[target] += 1

print("Combined Counts:")
for cat in TARGET_CLASSES:
    print(f"{cat:15}: {counts[cat]}")
