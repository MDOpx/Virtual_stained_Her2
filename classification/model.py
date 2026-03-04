import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not available. Install with: pip install timm")

__all__ = ['ClassificationModel', 'TIMM_AVAILABLE', 'create_backbone']


def create_backbone(backbone_name, is_pretrained=False):
    if backbone_name == 'resnet18':
        return resnet18(pretrained=is_pretrained)
    elif backbone_name == 'resnet50':
        return resnet50(pretrained=is_pretrained)
    if TIMM_AVAILABLE:
        try:
            model = timm.create_model(backbone_name, pretrained=is_pretrained, num_classes=0)
            num_features = model.num_features if hasattr(model, 'num_features') else model.default_cfg.get('num_features', 512)
            class TimmWrapper(nn.Module):
                def __init__(self, model, num_features):
                    super().__init__()
                    self.backbone = model
                    self.num_features = num_features
                @property
                def in_features(self):
                    return self.num_features
            return TimmWrapper(model, num_features)
        except Exception as e:
            raise ValueError(f"Failed to create model '{backbone_name}': {e}")
    raise ValueError(f"Model '{backbone_name}' requires timm. Install with: pip install timm")


class ClassificationModel(nn.Module):
    def __init__(self, num_classes=4, input_mode='B', backbone='resnet18', is_pretrained=False,
                 ab_fusion_mode='concat', ab_weight_A=1.0, ab_weight_B=1.0):
        super(ClassificationModel, self).__init__()
        self.input_mode = input_mode
        self.ab_fusion_mode = ab_fusion_mode if input_mode == 'AB' else None
        self.ab_weight_A = ab_weight_A if input_mode == 'AB' else 1.0
        self.ab_weight_B = ab_weight_B if input_mode == 'AB' else 1.0
        base_model = create_backbone(backbone, is_pretrained)
        if hasattr(base_model, 'fc'):
            num_features = base_model.fc.in_features
        elif hasattr(base_model, 'num_features'):
            num_features = base_model.num_features
        else:
            num_features = getattr(base_model, 'num_features', 512)
        if input_mode == 'AB':
            if hasattr(base_model, 'backbone'):
                self.feature_extractor_A = base_model.backbone
                self.feature_extractor_B = create_backbone(backbone, is_pretrained).backbone
            else:
                self.feature_extractor_A = nn.Sequential(*list(base_model.children())[:-1])
                base_model_B = create_backbone(backbone, is_pretrained)
                self.feature_extractor_B = nn.Sequential(*list(base_model_B.children())[:-1])
            if ab_fusion_mode in ('concat', 'weighted_concat'):
                self.fc = nn.Linear(num_features * 2, num_classes)
            elif ab_fusion_mode == 'weighted_sum':
                self.fc = nn.Linear(num_features, num_classes)
            else:
                raise ValueError(f"Unknown ab_fusion_mode: {ab_fusion_mode}")
        else:
            if hasattr(base_model, 'backbone'):
                self.model = base_model.backbone
                self.fc = nn.Linear(num_features, num_classes)
            else:
                base_model.fc = nn.Linear(num_features, num_classes)
                self.model = base_model
                self.fc = None

    def forward(self, x):
        if self.input_mode == 'AB':
            x_A, x_B = x[:, :3, :, :], x[:, 3:, :, :]
            feat_A = self.feature_extractor_A(x_A)
            feat_B = self.feature_extractor_B(x_B)
            if len(feat_A.shape) > 2:
                feat_A = feat_A.view(feat_A.size(0), -1)
            if len(feat_B.shape) > 2:
                feat_B = feat_B.view(feat_B.size(0), -1)
            if self.ab_fusion_mode in ('weighted_concat', 'weighted_sum'):
                feat_A, feat_B = feat_A * self.ab_weight_A, feat_B * self.ab_weight_B
            if self.ab_fusion_mode in ('concat', 'weighted_concat'):
                feat_combined = torch.cat([feat_A, feat_B], dim=1)
            else:
                feat_combined = feat_A + feat_B
            return self.fc(feat_combined)
        else:
            if self.fc is not None:
                feat = self.model(x)
                if len(feat.shape) > 2:
                    feat = feat.view(feat.size(0), -1)
                return self.fc(feat)
            return self.model(x)
