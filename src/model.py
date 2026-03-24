import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS      = torch.cuda.device_count()
NUM_CLASSES = 5
IMG_SIZE    = 512


class AttentionFusion(nn.Module):
    def __init__(self, feat_dims: list, out_dim: int = 512):
        super().__init__()
        total_dim = sum(feat_dims)   # 1792 + 2048 + 768 = 4608

        self.attention = nn.Sequential(
            nn.Linear(total_dim, len(feat_dims)),
            nn.Softmax(dim=1)
        )

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU()
            ) for d in feat_dims
        ])

        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, features: list):
        concat  = torch.cat(features, dim=1)
        weights = self.attention(concat)
        projected = [self.proj[i](features[i]) for i in range(len(features))]
        weighted  = sum(weights[:, i:i+1] * projected[i]
                        for i in range(len(features)))
        return self.fusion(weighted)


class DRClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.5,   # ✅ FIX — increased from 0.4 to 0.5
                 pretrained: bool = True):
        super().__init__()

        self.efficientnet = timm.create_model(
            'efficientnet_b4',
            pretrained  = pretrained,
            num_classes = 0,
            global_pool = 'avg'
        )

        self.resnet = timm.create_model(
            'resnet50',
            pretrained  = pretrained,
            num_classes = 0,
            global_pool = 'avg'
        )

        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained  = pretrained,
            num_classes = 0,
            img_size    = 224
        )

        self.feat_dims = [1792, 2048, 768]
        self.fusion    = AttentionFusion(self.feat_dims, out_dim=512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),   # ✅ now 0.5
            nn.Linear(128, num_classes)
        )

        self.gradients   = {}
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        def make_hooks(name):
            def forward_hook(module, input, output):
                self.activations[name] = output.detach()
            def backward_hook(module, grad_in, grad_out):
                self.gradients[name] = grad_out[0].detach()
            return forward_hook, backward_hook

        fwd, bwd = make_hooks('efficientnet')
        self.efficientnet.conv_head.register_forward_hook(fwd)
        self.efficientnet.conv_head.register_full_backward_hook(bwd)

        fwd, bwd = make_hooks('resnet')
        self.resnet.layer4[-1].register_forward_hook(fwd)
        self.resnet.layer4[-1].register_full_backward_hook(bwd)

    def get_efficientnet_features(self, x):
        return self.efficientnet(x)

    def get_resnet_features(self, x):
        return self.resnet(x)

    def get_vit_features(self, x):
        x_small = F.interpolate(x, size=(224, 224), mode='bilinear',
                                align_corners=False)
        return self.vit(x_small)

    def forward(self, x):
        f1     = self.get_efficientnet_features(x)
        f2     = self.get_resnet_features(x)
        f3     = self.get_vit_features(x)
        fused  = self.fusion([f1, f2, f3])
        logits = self.classifier(fused)
        return logits

    def get_attention_weights(self, x):
        f1 = self.get_efficientnet_features(x)
        f2 = self.get_resnet_features(x)
        f3 = self.get_vit_features(x)
        concat  = torch.cat([f1, f2, f3], dim=1)
        weights = self.fusion.attention(concat)
        return weights


class DRClassifierLite(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=pretrained,
            num_classes=0, global_pool='avg'
        )
        self.classifier = nn.Sequential(
            nn.Linear(1792, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


def build_model(arch: str = 'ensemble', pretrained: bool = True) -> nn.Module:
    if arch == 'ensemble':
        model = DRClassifier(pretrained=pretrained)
    elif arch == 'lite':
        model = DRClassifierLite(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown arch: {arch}. Use 'ensemble' or 'lite'.")

    model = model.to(DEVICE)

    if N_GPUS > 1:
        print(f"Multi-GPU: wrapping in DataParallel across {N_GPUS} GPUs")
        model = nn.DataParallel(model)

    return model
