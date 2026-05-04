import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ==================== MODEL ====================
def build_model(num_classes,
                device,
                ckpt_path=None,
                model_name='tf_efficientnet_b3.ns_jft_in1k',
                pretrained=False):

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)

        # 兼容 DataParallel
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}

        model.load_state_dict(state)
        print(f"Loaded checkpoint → {ckpt_path}")

    return model.to(device)


# ==================== FOCAL LOSS ====================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss