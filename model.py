import torch
import torch.nn as nn
from transformers import SiglipVisionModel


class GameUIModel(nn.Module):
    def __init__(
        self,
        model_name="google/siglip2-base-patch16-224",
        num_primary_screen_types=12,
        num_visual_style_tags=12,
        num_theme_tags=12,
        num_layout_positions=12,
        num_layout_element_types=12,
        num_layout_roles=12,
        freeze_backbone=True,
        dropout=0.1,
    ):
        super().__init__()

        print(f"[*] Loading backbone: {model_name}")
        self.backbone = SiglipVisionModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Single-label classification head.
        self.primary_screen_head = nn.Linear(
            self.hidden_size,
            num_primary_screen_types,
        )

        # Multi-label classification head for Visual Styles.
        self.visual_style_head = nn.Linear(
            self.hidden_size,
            num_visual_style_tags,
        )

        # Multi-label classification head for Themes.
        self.theme_head = nn.Linear(
            self.hidden_size,
            num_theme_tags,
        )

        # Multi-label classification heads for Layout (Split axes).
        self.layout_position_head = nn.Linear(
            self.hidden_size,
            num_layout_positions,
        )
        self.layout_element_type_head = nn.Linear(
            self.hidden_size,
            num_layout_element_types,
        )
        self.layout_role_head = nn.Linear(
            self.hidden_size,
            num_layout_roles,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[+] Backbone frozen.")

    def unfreeze_backbone(self, last_n_layers=2):
        if last_n_layers <= 0:
            self.freeze_backbone()
            return

        for param in self.backbone.parameters():
            param.requires_grad = True

        layers = self.backbone.vision_model.encoder.layers
        total_layers = len(layers)

        if last_n_layers < total_layers:
            frozen_count = total_layers - last_n_layers
            for i in range(frozen_count):
                for param in layers[i].parameters():
                    param.requires_grad = False

            for param in self.backbone.vision_model.embeddings.parameters():
                param.requires_grad = False

            print(f"[+] Last {last_n_layers} encoder layers unfrozen.")
        else:
            print("[+] Full backbone unfrozen.")

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        features = self.dropout(pooled_output)

        logits_primary_screen_type = self.primary_screen_head(features)
        logits_visual_style_tags = self.visual_style_head(features)
        logits_theme_tags = self.theme_head(features)
        logits_layout_positions = self.layout_position_head(features)
        logits_layout_element_types = self.layout_element_type_head(features)
        logits_layout_roles = self.layout_role_head(features)

        return {
            "logits_primary_screen_type": logits_primary_screen_type,
            "logits_visual_style_tags": logits_visual_style_tags,
            "logits_theme_tags": logits_theme_tags,
            "logits_layout_positions": logits_layout_positions,
            "logits_layout_element_types": logits_layout_element_types,
            "logits_layout_roles": logits_layout_roles,
        }


if __name__ == "__main__":
    model = GameUIModel(
        num_primary_screen_types=12,
        num_visual_style_tags=10,
        num_theme_tags=8,
        num_layout_positions=12,
        num_layout_element_types=15,
        num_layout_roles=10,
        freeze_backbone=True,
    )
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Output shapes:")
    print(f"Primary screen: {output['logits_primary_screen_type'].shape}")
    print(f"Visual Style: {output['logits_visual_style_tags'].shape}")
    print(f"Theme tags: {output['logits_theme_tags'].shape}")
    print(f"Layout Positions: {output['logits_layout_positions'].shape}")
    print(f"Layout Elements: {output['logits_layout_element_types'].shape}")
    print(f"Layout Roles: {output['logits_layout_roles'].shape}")
