# -*- coding: utf-8 -*-
"""
Training script for Game UI schema v6.0.

Targets:
- primary_screen_type: single-label classification
- style_tags: multi-label classification
- layout_tokens: multi-label classification

Expected trainable metadata rows:
- review_status == labeled
- is_game_ui == true
- ui_quality == keep
- primary_screen_type is not empty
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoProcessor

from dataset import GameUIDataset
from model import GameUIModel
import layout_vocab
from layout_vocab import POSITIONS, ELEMENT_TYPES, ROLES, normalize_layout_value


REQUIRED_COLUMNS = [
    "file_name",
    "is_game_ui",
    "ui_quality",
    "primary_screen_type",
    "visual_style_tags",
    "theme_tags",
    "layout_blocks",
    "layout_tokens",
    "components",
    "review_status",
]




def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def parse_json_list(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, dict):
                        result.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
                    else:
                        item_text = str(item).strip()
                        if item_text:
                            result.append(item_text)
                return result
        except Exception:
            pass

    return [item.strip() for item in text.split(",") if item.strip()]


def validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Invalid metadata schema. Missing columns: " + ", ".join(missing)
        )


def filter_trainable_rows(df: pd.DataFrame) -> pd.DataFrame:
    validate_schema(df)

    filtered = df.copy()
    filtered["_is_game_ui_bool"] = filtered["is_game_ui"].apply(parse_bool)
    filtered["_review_status_norm"] = filtered["review_status"].fillna("").astype(str).str.strip()
    filtered["_ui_quality_norm"] = filtered["ui_quality"].fillna("").astype(str).str.strip()
    filtered["_primary_norm"] = filtered["primary_screen_type"].fillna("").astype(str).str.strip()

    mask = (
        filtered["_is_game_ui_bool"]
        & (filtered["_review_status_norm"].isin(["labeled", "reviewed"]))
        & (filtered["_ui_quality_norm"] == "keep")
        & (filtered["_primary_norm"] != "")
    )

    return filtered[mask].copy().reset_index(drop=True)


def get_vocabs(csv_file: str, use_grouped: bool = False) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], pd.DataFrame]:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Metadata file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df = filter_trainable_rows(df)

    if len(df) == 0:
        raise ValueError("No trainable rows found. Label data first with labeling_tool.py.")

    # 1. Screen Types & Tags
    screen_col = "grouped_primary_screen" if use_grouped else "primary_screen_type"
    primary_screen_types = sorted({str(v).strip() for v in df[screen_col].dropna() if str(v).strip()})
    fallback_cls = "flow_other" if use_grouped else "other"
    if fallback_cls not in primary_screen_types: primary_screen_types.append(fallback_cls)

    style_col = "grouped_visual_style" if use_grouped else "visual_style_tags"
    v_style_set = set()
    for v in df[style_col].dropna():
        for tag in parse_json_list(v): v_style_set.add(tag)
    visual_style_tags = sorted(v_style_set)

    theme_set = set()
    for v in df["theme_tags"].dropna():
        for tag in parse_json_list(v): theme_set.add(tag)
    theme_tags = sorted(theme_set)

    # 2. Layout Vocab (Fixed standard)
    if use_grouped:
        layout_positions = ["top", "bottom", "center", "left", "right", "full_screen"]
        layout_element_types = ["panel_menu", "container_grid", "button", "preview_avatar", "text_box", "hud_bar_indicator"]
        layout_roles = ["navigation_quest", "status_resource", "combat_skill", "inventory_shop", "character_select", "system_narrative"]
    else:
        layout_positions = POSITIONS[:]
        layout_element_types = ELEMENT_TYPES[:]
        layout_roles = ROLES[:]

    # 3. Stats for Normalization
    from collections import Counter
    norm_stats = {
        "position": {"total": 0, "matched": 0, "defaulted": Counter()},
        "element_type": {"total": 0, "matched": 0, "defaulted": Counter()},
        "role": {"total": 0, "matched": 0, "defaulted": Counter()},
    }

    layout_col = "grouped_layout_blocks" if use_grouped else "layout_blocks"
    for value in df[layout_col].dropna():
        blocks = []
        try:
            text = str(value).strip()
            if text.startswith("[") and text.endswith("]"):
                blocks = json.loads(text.replace("'", '"'))
        except Exception: pass
        
        for b in blocks:
            if not isinstance(b, dict): continue
            for axis, allowed, default in [("position", layout_positions, "center"), ("element_type", layout_element_types, "panel"), ("role", layout_roles, "unknown")]:
                raw_v = b.get(axis, "")
                if use_grouped:
                    norm_v = str(raw_v).strip()
                else:
                    norm_v = normalize_layout_value(raw_v, axis)
                norm_stats[axis]["total"] += 1
                
                # Check if it's a match
                if norm_v in allowed:
                    norm_stats[axis]["matched"] += 1
                else:
                    norm_stats[axis]["defaulted"][str(raw_v)] += 1

    print("\n" + "="*50)
    print(" [Layout Normalization Statistics]")
    for axis in ["position", "element_type", "role"]:
        s = norm_stats[axis]
        print(f" * {axis.upper()}: Std Vocab {len(getattr(layout_vocab, axis.upper()+'S')) if hasattr(layout_vocab, axis.upper()+'S') and not use_grouped else len(allowed)}, Total Instances {s['total']}")
        print(f"   - Matched: {s['matched']} ({s['matched']/max(1,s['total']):.1%})")
        print(f"   - Defaulted: {sum(s['defaulted'].values())} ({sum(s['defaulted'].values())/max(1,s['total']):.1%})")
        if s['defaulted']:
            print(f"   - Top 10 Defaulted Raw Labels: {s['defaulted'].most_common(10)}")
    print("="*50 + "\n")

    return primary_screen_types, visual_style_tags, theme_tags, layout_positions, layout_element_types, layout_roles, df



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_weighted_sampler(full_dataset: GameUIDataset, train_indices: List[int], use_grouped: bool = False) -> WeightedRandomSampler:
    train_df = full_dataset.df.iloc[train_indices]
    screen_col = "grouped_primary_screen" if use_grouped else "primary_screen_type"
    counts = train_df[screen_col].value_counts().to_dict()
    class_weights = {label: 1.0 / count for label, count in counts.items() if count > 0}
    sample_weights = [
        class_weights.get(full_dataset.df.iloc[i][screen_col], 1.0)
        for i in train_indices
    ]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def compute_multilabel_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "precision": precision_score(gt, pred, average="macro", zero_division=0),
        "recall": recall_score(gt, pred, average="macro", zero_division=0),
        "f1": f1_score(gt, pred, average="macro", zero_division=0),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    print(f"[*] Using device: {device}")
    print(f"[*] AMP enabled: {amp_enabled}")
    print(f"[*] Loss weights: style={args.style_loss_weight}, layout={args.layout_loss_weight}")
    print(f"[*] Train config: lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}, unfreeze_last_n={args.unfreeze_last_n}")

    try:
        primary_screen_types, visual_style_tags, theme_tags, layout_positions, layout_element_types, layout_roles, df_clean = get_vocabs(args.csv_file, use_grouped=args.grouped)
    except ValueError as e:
        print("=" * 60)
        print(f"[!] {e}")
        print("=" * 60)
        return
    print(
        "[*] Vocab: "
        f"{len(primary_screen_types)} primary screen types, "
        f"{len(visual_style_tags)} visual style tags, "
        f"{len(theme_tags)} theme tags, "
        f"{len(layout_positions)} layout positions, "
        f"{len(layout_element_types)} layout elements, "
        f"{len(layout_roles)} layout roles"
    )
    print(f"[*] Trainable rows in metadata: {len(df_clean)}")

    model_name = args.model_name
    processor = AutoProcessor.from_pretrained(model_name)

    full_dataset = GameUIDataset(
        csv_file=args.csv_file,
        img_dir=args.img_dir,
        processor=processor,
        primary_screen_types=primary_screen_types,
        visual_style_tags=visual_style_tags,
        theme_tags=theme_tags,
        layout_positions=layout_positions,
        layout_element_types=layout_element_types,
        layout_roles=layout_roles,
        use_grouped=args.grouped,
    )

    if len(full_dataset) < 5:
        print("=" * 60)
        print("[!] Too few labeled samples to proceed with training.")
        print(f"[*] Required: at least 5 'keep' samples with 'labeled' status.")
        print(f"[*] Found: {len(full_dataset)}")
        print("=" * 60)
        return

    # Stable split using is_val_split column if present
    if "is_val_split" in full_dataset.df.columns:
        val_indices = full_dataset.df[full_dataset.df["is_val_split"] == True].index.tolist()
        train_indices = full_dataset.df[full_dataset.df["is_val_split"] == False].index.tolist()
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        train_size = len(train_indices)
        val_size = len(val_indices)
        print(f"[*] Loaded custom stable split: Train {train_size}, Val {val_size}")
    else:
        val_ratio = args.val_ratio
        val_size = int(len(full_dataset) * val_ratio)
        val_size = max(1, min(val_size, len(full_dataset) - 1))
        train_size = len(full_dataset) - val_size

        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=generator,
        )
        print(f"[*] Data split (fallback): Train {train_size}, Val {val_size}")

    sampler = make_weighted_sampler(full_dataset, list(train_dataset.indices), use_grouped=args.grouped)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = GameUIModel(
        model_name=model_name,
        num_primary_screen_types=len(primary_screen_types),
        num_visual_style_tags=len(visual_style_tags),
        num_theme_tags=len(theme_tags),
        num_layout_positions=len(layout_positions),
        num_layout_element_types=len(layout_element_types),
        num_layout_roles=len(layout_roles),
        freeze_backbone=True,
        dropout=args.dropout,
    )
    if args.unfreeze_last_n > 0:
        model.unfreeze_backbone(args.unfreeze_last_n)
    model.to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=1,
        factor=0.5,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if args.weighted_loss:
        train_df = full_dataset.df.iloc[train_dataset.indices]
        screen_col = "grouped_primary_screen" if args.grouped else "primary_screen_type"
        counts = train_df[screen_col].value_counts().to_dict()
        total_train = len(train_df)
        num_classes = len(primary_screen_types)
        class_weights = []
        for cls in primary_screen_types:
            count = counts.get(cls, 0)
            weight = total_train / (num_classes * count) if count > 0 else 1.0
            class_weights.append(weight)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
        criterion_primary = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(f"[*] Applied CrossEntropyLoss weights: {dict(zip(primary_screen_types, class_weights))}")
    else:
        criterion_primary = nn.CrossEntropyLoss()
    criterion_visual_style = nn.BCEWithLogitsLoss()
    criterion_theme = nn.BCEWithLogitsLoss()
    criterion_layout_pos = nn.BCEWithLogitsLoss()
    criterion_layout_elem = nn.BCEWithLogitsLoss()
    criterion_layout_role = nn.BCEWithLogitsLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    full_dataset.save_vocab(os.path.join(args.output_dir, "vocab.json"))

    best_score = -1.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_primary_loss = 0.0
        train_visual_style_loss = 0.0
        train_theme_loss = 0.0
        train_layout_pos_loss = 0.0
        train_layout_elem_loss = 0.0
        train_layout_role_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]"):
            optimizer.zero_grad(set_to_none=True)

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            primary_labels = batch["primary_screen_label"].to(device, non_blocking=True)
            visual_style_labels = batch["visual_style_label"].to(device, non_blocking=True)
            theme_labels = batch["theme_label"].to(device, non_blocking=True)
            layout_pos_labels = batch["layout_position_label"].to(device, non_blocking=True)
            layout_elem_labels = batch["layout_element_type_label"].to(device, non_blocking=True)
            layout_role_labels = batch["layout_role_label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(pixel_values)
                loss_primary = criterion_primary(
                    outputs["logits_primary_screen_type"],
                    primary_labels,
                )
                loss_visual_style = criterion_visual_style(
                    outputs["logits_visual_style_tags"],
                    visual_style_labels,
                )
                loss_theme = criterion_theme(
                    outputs["logits_theme_tags"],
                    theme_labels,
                )
                loss_layout_pos = criterion_layout_pos(
                    outputs["logits_layout_positions"],
                    layout_pos_labels,
                )
                loss_layout_elem = criterion_layout_elem(
                    outputs["logits_layout_element_types"],
                    layout_elem_labels,
                )
                loss_layout_role = criterion_layout_role(
                    outputs["logits_layout_roles"],
                    layout_role_labels,
                )
                
                loss = (
                    loss_primary
                    + args.style_loss_weight * (loss_visual_style + loss_theme)
                    + args.layout_loss_weight * (loss_layout_pos + loss_layout_elem + loss_layout_role)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            train_primary_loss += float(loss_primary.item())
            train_visual_style_loss += float(loss_visual_style.item())
            train_theme_loss += float(loss_theme.item())
            train_layout_pos_loss += float(loss_layout_pos.item())
            train_layout_elem_loss += float(loss_layout_elem.item())
            train_layout_role_loss += float(loss_layout_role.item())

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_train_primary_loss = train_primary_loss / max(1, len(train_loader))
        avg_train_visual_style_loss = train_visual_style_loss / max(1, len(train_loader))
        avg_train_theme_loss = train_theme_loss / max(1, len(train_loader))
        avg_train_layout_pos_loss = train_layout_pos_loss / max(1, len(train_loader))
        avg_train_layout_elem_loss = train_layout_elem_loss / max(1, len(train_loader))
        avg_train_layout_role_loss = train_layout_role_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_primary_loss = 0.0
        val_visual_style_loss = 0.0
        val_theme_loss = 0.0
        val_layout_pos_loss = 0.0
        val_layout_elem_loss = 0.0
        val_layout_role_loss = 0.0

        all_primary_preds: List[int] = []
        all_primary_gt: List[int] = []
        all_v_style_preds: List[np.ndarray] = []
        all_v_style_gt: List[np.ndarray] = []
        all_theme_preds: List[np.ndarray] = []
        all_theme_gt: List[np.ndarray] = []
        all_layout_pos_preds: List[np.ndarray] = []
        all_layout_pos_gt: List[np.ndarray] = []
        all_layout_elem_preds: List[np.ndarray] = []
        all_layout_elem_gt: List[np.ndarray] = []
        all_layout_role_preds: List[np.ndarray] = []
        all_layout_role_gt: List[np.ndarray] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]"):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                primary_labels = batch["primary_screen_label"].to(device, non_blocking=True)
                visual_style_labels = batch["visual_style_label"].to(device, non_blocking=True)
                theme_labels = batch["theme_label"].to(device, non_blocking=True)
                layout_pos_labels = batch["layout_position_label"].to(device, non_blocking=True)
                layout_elem_labels = batch["layout_element_type_label"].to(device, non_blocking=True)
                layout_role_labels = batch["layout_role_label"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(pixel_values)
                    loss_primary = criterion_primary(
                        outputs["logits_primary_screen_type"],
                        primary_labels,
                    )
                    loss_visual_style = criterion_visual_style(
                        outputs["logits_visual_style_tags"],
                        visual_style_labels,
                    )
                    loss_theme = criterion_theme(
                        outputs["logits_theme_tags"],
                        theme_labels,
                    )
                    loss_layout_pos = criterion_layout_pos(
                        outputs["logits_layout_positions"],
                        layout_pos_labels,
                    )
                    loss_layout_elem = criterion_layout_elem(
                        outputs["logits_layout_element_types"],
                        layout_elem_labels,
                    )
                    loss_layout_role = criterion_layout_role(
                        outputs["logits_layout_roles"],
                        layout_role_labels,
                    )
                    loss = (
                        loss_primary
                        + args.style_loss_weight * (loss_visual_style + loss_theme)
                        + args.layout_loss_weight * (loss_layout_pos + loss_layout_elem + loss_layout_role)
                    )

                val_loss += float(loss.item())
                val_primary_loss += float(loss_primary.item())
                val_visual_style_loss += float(loss_visual_style.item())
                val_theme_loss += float(loss_theme.item())
                val_layout_pos_loss += float(loss_layout_pos.item())
                val_layout_elem_loss += float(loss_layout_elem.item())
                val_layout_role_loss += float(loss_layout_role.item())

                primary_preds = torch.argmax(outputs["logits_primary_screen_type"], dim=1)
                v_style_preds = (torch.sigmoid(outputs["logits_visual_style_tags"]) > args.style_threshold).float()
                theme_preds = (torch.sigmoid(outputs["logits_theme_tags"]) > args.style_threshold).float()
                layout_pos_preds = (torch.sigmoid(outputs["logits_layout_positions"]) > args.layout_threshold).float()
                layout_elem_preds = (torch.sigmoid(outputs["logits_layout_element_types"]) > args.layout_threshold).float()
                layout_role_preds = (torch.sigmoid(outputs["logits_layout_roles"]) > args.layout_threshold).float()

                all_primary_preds.extend(primary_preds.cpu().numpy().tolist())
                all_primary_gt.extend(primary_labels.cpu().numpy().tolist())
                all_v_style_preds.append(v_style_preds.cpu().numpy())
                all_v_style_gt.append(visual_style_labels.cpu().numpy())
                all_theme_preds.append(theme_preds.cpu().numpy())
                all_theme_gt.append(theme_labels.cpu().numpy())
                all_layout_pos_preds.append(layout_pos_preds.cpu().numpy())
                all_layout_pos_gt.append(layout_pos_labels.cpu().numpy())
                all_layout_elem_preds.append(layout_elem_preds.cpu().numpy())
                all_layout_elem_gt.append(layout_elem_labels.cpu().numpy())
                all_layout_role_preds.append(layout_role_preds.cpu().numpy())
                all_layout_role_gt.append(layout_role_labels.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        primary_acc = accuracy_score(all_primary_gt, all_primary_preds)

        v_style_pred_np = np.vstack(all_v_style_preds)
        v_style_gt_np = np.vstack(all_v_style_gt)
        theme_pred_np = np.vstack(all_theme_preds)
        theme_gt_np = np.vstack(all_theme_gt)

        v_style_metrics = compute_multilabel_metrics(v_style_gt_np, v_style_pred_np)
        theme_metrics = compute_multilabel_metrics(theme_gt_np, theme_pred_np)
        
        layout_pos_pred_np = np.vstack(all_layout_pos_preds)
        layout_pos_gt_np = np.vstack(all_layout_pos_gt)
        layout_elem_pred_np = np.vstack(all_layout_elem_preds)
        layout_elem_gt_np = np.vstack(all_layout_elem_gt)
        layout_role_pred_np = np.vstack(all_layout_role_preds)
        layout_role_gt_np = np.vstack(all_layout_role_gt)
        
        layout_pos_metrics = compute_multilabel_metrics(layout_pos_gt_np, layout_pos_pred_np)
        layout_elem_metrics = compute_multilabel_metrics(layout_elem_gt_np, layout_elem_pred_np)
        layout_role_metrics = compute_multilabel_metrics(layout_role_gt_np, layout_role_pred_np)

        layout_avg_f1 = (layout_pos_metrics["f1"] + layout_elem_metrics["f1"] + layout_role_metrics["f1"]) / 3.0

        selection_score = (
            primary_acc
            + v_style_metrics["f1"]
            + theme_metrics["f1"]
            + layout_avg_f1
        ) / 4.0

        print(f"[*] Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")
        print(f"[*] Pri Acc: {primary_acc:.4f} | V-Style F1: {v_style_metrics['f1']:.4f} | Theme F1: {theme_metrics['f1']:.4f}")
        print(f"[*] Layout Pos F1: {layout_pos_metrics['f1']:.4f} | Elem F1: {layout_elem_metrics['f1']:.4f} | Role F1: {layout_role_metrics['f1']:.4f} | Layout Avg F1: {layout_avg_f1:.4f} | Score: {selection_score:.4f}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Metric/Primary_Acc", primary_acc, epoch)
        writer.add_scalar("Metric/VStyle_F1", v_style_metrics["f1"], epoch)
        writer.add_scalar("Metric/Theme_F1", theme_metrics["f1"], epoch)
        writer.add_scalar("Metric/Layout_Position_F1", layout_pos_metrics["f1"], epoch)
        writer.add_scalar("Metric/Layout_Element_F1", layout_elem_metrics["f1"], epoch)
        writer.add_scalar("Metric/Layout_Role_F1", layout_role_metrics["f1"], epoch)
        writer.add_scalar("Metric/Layout_Avg_F1", layout_avg_f1, epoch)

        scheduler.step(selection_score)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_score": best_score,
            "vocab": {
                "primary_screen_types": primary_screen_types,
                "visual_style_tags": visual_style_tags,
                "theme_tags": theme_tags,
                "layout_positions": layout_positions,
                "layout_element_types": layout_element_types,
                "layout_roles": layout_roles,
            },
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last_model.pth"))

        if selection_score > best_score:
            best_score = selection_score
            patience_counter = 0
            torch.save(ckpt, os.path.join(args.output_dir, "best_model.pth"))
            print(f"[+] Best model saved with score: {best_score:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"[*] Early stopping triggered.")
            break

    writer.close()

    # Create reports dir
    os.makedirs("reports", exist_ok=True)

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_primary_gt, all_primary_preds)
    cm_df = pd.DataFrame(cm, index=primary_screen_types, columns=primary_screen_types)
    cm_df.to_csv("reports/retrain_midcheck_confusion_matrix.csv")

    # Class Metrics
    report_dict = classification_report(all_primary_gt, all_primary_preds, target_names=primary_screen_types, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("reports/retrain_midcheck_class_metrics.csv")

    # Final summary CSV
    summary_df = pd.DataFrame({
        "best_score": [best_score],
        "primary_acc": [primary_acc],
        "v_style_f1": [v_style_metrics["f1"]],
        "theme_f1": [theme_metrics["f1"]],
        "layout_avg_f1": [layout_avg_f1]
    })
    summary_df.to_csv("reports/retrain_midcheck_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="data/metadata.csv")
    parser.add_argument("--img_dir", type=str, default="data/images")
    parser.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze_last_n", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--style_threshold", type=float, default=0.5)
    parser.add_argument("--layout_threshold", type=float, default=0.5)
    parser.add_argument("--style_loss_weight", type=float, default=1.0)
    parser.add_argument("--layout_loss_weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--grouped", action="store_true", help="Use grouped labels for training")
    parser.add_argument("--weighted_loss", action="store_true", help="Apply class-weighted loss for primary screen type")

    train(parser.parse_args())
