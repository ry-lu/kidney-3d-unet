
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    

class BCELoss(nn.Module):
    def __init__(self, smooth=1.0, pos_weight=1, device='cpu'):
        super(BCELoss, self).__init__()
        self.smooth = smooth
        self.pos_weight = pos_weight
        self.device = device

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        pos_weight = torch.tensor([self.pos_weight]).to(self.device)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,pos_weight=pos_weight)
        return loss
    

class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=0, size_average=None, ignore_index=-100,
                 reduce=None, balance_param=1.0):
        super(FocalLoss, self).__init__(size_average)
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        logpt = - F.binary_cross_entropy_with_logits(input, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, pos_weight=1, device='cpu'):
        super(DiceBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.device = device
        
    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        pos_weight = torch.tensor([self.pos_weight]).to(self.device)
        bce_loss = F.binary_cross_entropy(inputs, targets, pos_weight=pos_weight,reduction='mean')
        Dice_BCE = (bce_loss + dice_loss)/2
        
        return Dice_BCE