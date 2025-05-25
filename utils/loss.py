import torch
from torch import nn
from torchvision.models.vgg import vgg16, VGG16_Weights

# Custom Loss function that combines Huber loss with total variance and
# VGG16 perceptual loss
class UVSVENetLoss(nn.Module):
    def __init__(self):
        super(UVSVENetLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        vgg16_net = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in vgg16_net.parameters():
            param.requires_grad = False
        # VGG 16 loss network
        self.vgg16 = vgg16_net
        # L2 MSE Loss
        self.l2 = nn.MSELoss()
        # L1 MAE Loss
        self.l1 = nn.L1Loss()
        # Total Variance loss
        self.loss_total_variance = TotalVarianceLoss()
        # Previous gradients for temporal consistency loss
        self.previous_vgg_features = None

    def forward(self, out_images, target_images):
        # Calculate VGG features for current predictions separately
        # for reuse in temporal loss
        current_vgg_features = self.vgg16(out_images)
        loss_perception = self.l2(current_vgg_features, self.vgg16(target_images))

        # Do not calculate temporal loss for the first frame
        if self.previous_vgg_features is not None:
            # We can also use L1 or Huber loss for this
            loss_temporal = self.l2(self.previous_vgg_features, current_vgg_features)
        else:
            loss_temporal = 0
        
        # Huber loss using a ratio of 3:2 for L1 and L2 losses
        loss_reconstruction = (
            0.6 * self.l1(out_images, target_images) + 
            0.4 * self.l2(out_images, target_images)
        )
        loss_total_variance = self.loss_total_variance(out_images)

        # We can tweak these weights, but always make sure that
        # loss_reconstruction has the highest weight
        loss_final = (
            1 * loss_reconstruction + 
            0.01 * loss_perception + 
            2e-8 * loss_total_variance + 
            0.006 * loss_temporal
        )
        # Important so that we don't have null pointers stored
        self.previous_vgg_features = current_vgg_features.detach()

        return loss_final

# Total Variance Loss function
class TotalVarianceLoss(nn.Module):
    def __init__(self):
        super(TotalVarianceLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = get_size(x[:, :, 1:, :])
        count_w = get_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

# Get the number of elements in a tensor
def get_size(tensor):
    return tensor.size()[1] * tensor.size()[2] * tensor.size()[3]