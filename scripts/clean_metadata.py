import pandas as pd
import os

METADATA_FILE = "data/metadata.csv"
FINAL_COLUMNS = [
    "source_api", "moby_game_id", "game_title", "moby_url", "platform_id",
    "platform_name", "screenshot_caption", "screenshot_url", "file_name",
    "image_hash", "genre", "collected_at", "schema_version", "is_game_ui",
    "ui_quality", "primary_screen_type", "secondary_screen_types",
    "visual_style_tags", "theme_tags", "layout_blocks", "layout_tokens",
    "components", "confidence", "needs_review", "review_status",
    "ui_score", "ui_score_reason", "notes"
]

if os.path.exists(METADATA_FILE):
    df = pd.read_csv(METADATA_FILE, dtype=str).fillna("")
    # Reorder and filter
    # If some columns are missing, they will be created as empty strings
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    
    # Drop columns not in FINAL_COLUMNS (like legacy style_tags)
    df = df[FINAL_COLUMNS]
    
    df.to_csv(METADATA_FILE, index=False, encoding="utf-8-sig")
    print(f"Cleaned {METADATA_FILE}. Current columns: {list(df.columns)}")
else:
    print(f"{METADATA_FILE} not found.")
