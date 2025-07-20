import torch
import numpy as np
import math
import torch.nn.functional as F
import pickle 

class EntropyWeights:
    def __init__(self, tot_classes, old_classes) -> None:
        self.tot_classes = tot_classes + 1
        self.old_classes = old_classes + 1
        self.num = 0
        self.save_entropy = []
    
    def update_m(self, seg_label):
        seg_label = seg_label.clone().detach()
        seg_label[seg_label == 255] = 0
        B = seg_label.shape[0]

        histogram = torch.histc(seg_label.float(), bins=self.tot_classes)
        histogram = histogram / histogram.sum()

        if not hasattr(self, 'm'):
            self.m = histogram
        else:
            self.m = (self.m * self.num + histogram * B)/(self.num + B)
        self.num += B
    
    def get_num_weight(self, seg_label):
        num_weight = torch.zeros_like(seg_label, dtype=torch.float)

        class_num = (self.m > 0).to(int).sum()
        class_weight = self.m[2:].sum()/((class_num-2) * self.m)
        class_weight[0] = 1.
        class_weight[1] = 1.

        low = np.sqrt(float((self.tot_classes - self.old_classes) / self.old_classes))
        high = np.sqrt(float((self.old_classes) / (self.tot_classes - self.old_classes)))

        # low = 0.8; high = 1/0.8
         
        class_weight = torch.clamp(class_weight, min=low, max=high)

        for i in range(self.tot_classes):
            num_weight[seg_label == i] = class_weight[i]
        return num_weight
    
    def get_entropy_weight(self, seg_entropy, seg_label):
        entropy_high = 0.5
        seg_entropy[seg_label >= self.old_classes] = 0.
        scale = 1/(entropy_high + 1e-8)
        entropy_weight = (torch.exp(-scale * ((seg_entropy - entropy_high)))).to(float)
        entropy_weight[(seg_entropy >= entropy_high) & (seg_label < self.old_classes)] = 0.
        
        return entropy_weight

    def get_weight(self, seg_label, seg_output):
        seg_logit_softmax = F.softmax(seg_output, dim=1)
        seg_label = seg_label.clone().detach()
        seg_entropy = entropy(seg_logit_softmax)

        self.update_m(seg_label) 
        num_weight = self.get_num_weight(seg_label).to(torch.float32)
        entropy_weight = self.get_entropy_weight(seg_entropy, seg_label).to(torch.float32)

        weight =  entropy_weight * num_weight
        weight = weight / weight.mean()

        return weight

    def show_weight(self, seg_label, seg_output):
        seg_logit_softmax = F.softmax(seg_output, dim=1)
        seg_entropy = entropy(seg_logit_softmax).clone().detach()

        self.update_m(seg_label)
        num_weight = self.get_num_weight(seg_label).to(torch.float32)
        entropy_weight = self.get_entropy_weight(seg_entropy, seg_label).to(torch.float32)
        weight = num_weight * entropy_weight

        return weight, num_weight, entropy_weight, seg_entropy
    
    def update_weight(self, weight, seg_label):
        seg_label = seg_label.clone().detach()
        seg_label[seg_label== 255] =0
        weight = weight.clone().detach()
        B, H, W = seg_label.shape
        if not hasattr(self, 'weight_count'):
            self.weight_count = torch.zeros(self.tot_classes).to(seg_label.device)
            self.weight_num = 0
    
        for i in range(self.tot_classes):
            self.weight_count[i] = (weight[seg_label==i].sum()/weight.sum() * B + self.weight_count[i] * self.weight_num)/(self.weight_num + B)
        self.weight_num += B

    def entropy_statistics(self, gt_label, predit_label, seg_entropy, step, num_bins = 20):

        B,H,W = seg_entropy.shape

        gt_label = torch.tensor(gt_label).detach().to(seg_entropy.device)

        predit_label = torch.tensor(predit_label).clone().detach().to(seg_entropy.device)
        
        mask_bg = predit_label > 0

        gt_label = gt_label[mask_bg]
        predit_label = predit_label[mask_bg]
        seg_entropy = seg_entropy[mask_bg]

        bins = torch.linspace(0, 2, num_bins + 1).to(seg_entropy.device)
        bin_indices = torch.bucketize(seg_entropy, bins)
        right_indices = (gt_label == predit_label).to(int) * bin_indices 
        bins_count_all = torch.bincount(bin_indices.view(-1), minlength = (num_bins + 1))
        bins_count_right = torch.bincount((right_indices).view(-1), minlength = (num_bins + 1)) 
        bins_count_right[0] = 0.
        bins_count = bins_count_right / ( bins_count_all + 1e-8)
        entropy_num = bins_count_all / (B*H*W)

        if not hasattr(self, 'entropy_m'):
            self.entropy_m = bins_count
            self.entropy_num = entropy_num
            self.entropy_num_num = B
        else:
            self.entropy_m = (self.entropy_m * self.entropy_num * self.entropy_num_num + bins_count * entropy_num * B)/(self.entropy_num * self.entropy_num_num + entropy_num * B)
            self.entropy_num = (self.entropy_num  * self.entropy_num_num + entropy_num * B)/(self.entropy_num_num + B)
            self.entropy_num_num += B

        np.save('%s-entropy_m.npy'%(step), self.entropy_m.cpu().numpy())
        np.save('%s-entropy_num.npy'%(step), self.entropy_num.cpu().numpy()) 

def entropy(probabilities):
    """Computes the entropy per pixel.

    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
          Saporta et al.
          CVPR Workshop 2020

    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = probabilities.shape[1] / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)