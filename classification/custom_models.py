import torch
import torch.nn as nn
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point

class CustomPointCloudClassifier(nn.Module):
    def __init__(self, backbone_cfg, num_classes):
        super(CustomPointCloudClassifier, self).__init__()

        # Remove 'type' from backbone_cfg and set 'cls_mode' to True
        backbone_cfg_no_type = backbone_cfg.copy()
        backbone_cfg_no_type.pop('type', None)
        backbone_cfg_no_type['cls_mode'] = True

        # Build the backbone (encoder) with the adjusted configuration
        self.backbone = PointTransformerV3(**backbone_cfg_no_type)

        #Added for triplet Loss
        enc_dim = backbone_cfg_no_type["enc_channels"][-1]
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier  = nn.Sequential(
            nn.Linear(enc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(enc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        #Classifier Code
        '''
        self.num_classes = num_classes

        # Get the output channels from the backbone
        encoder_output_channels = backbone_cfg_no_type['enc_channels'][-1]

        # Define the global classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_channels, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        '''
    def forward(self, point: Point):
        # Run the backbone
        point = self.backbone(point)

        # Extract features
        x = point.feat  # Shape: [N, C]

        # Perform global pooling
        x = x.transpose(0, 1).unsqueeze(0)  # Shape: [1, C, N]
        x = self.global_pool(x).squeeze(2)   # Shape: [1, C]

        logits = self.classifier(x)        # classification branch

        out = {"cls_logits": logits,       # keep for multitask training
               "global_feat": x}           # expose pooled backbone feature

        # If an embedding head exists, produce an embedding
        if hasattr(self, "embedding_head"):
            emb = self.embedding_head(x)           # [1, embed_dim]
            emb = nn.functional.normalize(emb, p=2, dim=1)
            out["embedding"] = emb

        return out
