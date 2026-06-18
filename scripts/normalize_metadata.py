# -*- coding: utf-8 -*-
import pandas as pd
import json
import os
import shutil
from collections import Counter
from layout_vocab import normalize_layout_value, POSITIONS, ELEMENT_TYPES, ROLES

METADATA_PATH = "data/metadata.csv"
BACKUP_PATH = "data/metadata_raw_backup.csv"

def normalize_metadata():
    if not os.path.exists(METADATA_PATH):
        print(f"[Error] Metadata not found at {METADATA_PATH}")
        return

    # 1. Backup
    print(f"[*] Backing up {METADATA_PATH} to {BACKUP_PATH}")
    shutil.copy2(METADATA_PATH, BACKUP_PATH)

    # 2. Load
    df = pd.read_csv(METADATA_PATH)
    total_rows = len(df)
    
    # Stats
    stats = {
        "blocks_parsed": 0,
        "blocks_failed": 0,
        "pos_default": 0,
        "elem_default": 0,
        "role_default": 0,
        "default_labels": Counter()
    }

    def process_row(row):
        # Process layout_blocks
        blocks_json = row.get("layout_blocks")
        normalized_blocks = []
        if isinstance(blocks_json, str) and blocks_json.strip():
            try:
                # Handle single quotes if present
                blocks = json.loads(blocks_json.replace("'", '"'))
                if isinstance(blocks, list):
                    stats["blocks_parsed"] += 1
                    for b in blocks:
                        if not isinstance(b, dict): continue
                        
                        raw_pos = b.get("position", "")
                        raw_elem = b.get("element_type", "")
                        raw_role = b.get("role", "")
                        
                        norm_pos = normalize_layout_value(raw_pos, "position")
                        norm_elem = normalize_layout_value(raw_elem, "element_type")
                        norm_role = normalize_layout_value(raw_role, "role")
                        
                        if norm_pos == "center" and str(raw_pos).strip().lower() != "center":
                            stats["pos_default"] += 1
                            stats["default_labels"][f"pos:{raw_pos}"] += 1
                        if norm_elem == "panel" and str(raw_elem).strip().lower() != "panel":
                            stats["elem_default"] += 1
                            stats["default_labels"][f"elem:{raw_elem}"] += 1
                        if norm_role == "unknown" and str(raw_role).strip().lower() != "unknown":
                            stats["role_default"] += 1
                            stats["default_labels"][f"role:{raw_role}"] += 1
                            
                        normalized_blocks.append({
                            "position": norm_pos,
                            "element_type": norm_elem,
                            "role": norm_role
                        })
                else:
                    stats["blocks_failed"] += 1
            except Exception:
                stats["blocks_failed"] += 1
        
        # Update layout_blocks
        row["layout_blocks"] = json.dumps(normalized_blocks, ensure_ascii=False)
        
        # Re-generate layout_tokens from normalized blocks
        tokens = []
        for b in normalized_blocks:
            token = f"{b['position']}:{b['element_type']}:{b['role']}"
            tokens.append(token)
        row["layout_tokens"] = ", ".join(tokens)
        
        # Process components
        comp_raw = row.get("components")
        if isinstance(comp_raw, str) and comp_raw.strip():
            # If it's a list-like string, try to normalize each item as element_type
            try:
                if comp_raw.startswith("[") and comp_raw.endswith("]"):
                    comps = json.loads(comp_raw.replace("'", '"'))
                else:
                    comps = [c.strip() for c in comp_raw.split(",") if c.strip()]
                
                norm_comps = [normalize_layout_value(c, "element_type") for c in comps]
                row["components"] = ", ".join(norm_comps)
            except Exception:
                pass # Keep as is if failed
                
        return row

    print("[*] Normalizing rows...")
    df = df.apply(process_row, axis=1)

    # 3. Save
    df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")
    print(f"[+] Normalized metadata saved to {METADATA_PATH}")

    # 4. Validation Stats
    print("\n" + "="*50)
    print(" [Normalization Validation Results]")
    print(f" - Total Rows: {total_rows}")
    print(f" - Blocks Parsed: {stats['blocks_parsed']}")
    print(f" - Blocks Failed: {stats['blocks_failed']}")
    print(f" - Position Default Mappings: {stats['pos_default']}")
    print(f" - Element Type Default Mappings: {stats['elem_default']}")
    print(f" - Role Default Mappings: {stats['role_default']}")
    
    # Calculate unique counts from all rows
    all_pos = set()
    all_elem = set()
    all_role = set()
    for b_json in df["layout_blocks"]:
        try:
            blocks = json.loads(b_json)
            for b in blocks:
                all_pos.add(b["position"])
                all_elem.add(b["element_type"])
                all_role.add(b["role"])
        except: pass
        
    print(f" - Unique Positions: {len(all_pos)} {sorted(list(all_pos))}")
    print(f" - Unique Elements: {len(all_elem)}")
    print(f" - Unique Roles: {len(all_role)}")
    
    if stats["default_labels"]:
        print(f" - Top 10 Defaulted Raw Labels:")
        for label, count in stats["default_labels"].most_common(10):
            print(f"   {label}: {count}")
    print("="*50 + "\n")

if __name__ == "__main__":
    normalize_metadata()
