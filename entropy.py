import torch
import numpy as np
import math
import torch.nn.functional as F
import pickle 

class EntropyWeights:
    def __init__(self, tot_classes, old_classes) -> None:
        self.tot_classes = tot_classes
        self.old_classes = old_classes
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
        class_weight = self.m[1:].sum()/((class_num-1) * self.m)
        class_weight[0] = 1.

        low = np.sqrt(float((self.tot_classes - self.old_classes) / self.old_classes))
        high = np.sqrt(float((self.old_classes) / (self.tot_classes - self.old_classes)))
        
        class_weight = torch.clamp(class_weight, min=low, max=high)

        for i in range(self.tot_classes):
            num_weight[seg_label == i] = class_weight[i]
        return num_weight
    
    def get_entropy_weight(self, seg_entropy, seg_label):
        seg_entropy = (seg_entropy-seg_entropy.min())/(seg_entropy.max()-seg_entropy.min())
        entropy_high = 0.8
        seg_entropy[seg_label >= self.old_classes] = 0.
        scale = 1/(entropy_high + 1e-8)
        entropy_weight = (torch.exp(-scale * ((seg_entropy - entropy_high)))).to(float)
        entropy_weight[(seg_entropy >= entropy_high) & (seg_label < self.old_classes)] = 0.
        
        return entropy_weight

    def get_weight(self, seg_label, seg_output):
        seg_logit_softmax = F.softmax(seg_output, dim=1)
        seg_label = seg_label.clone().detach()
        seg_label[seg_label == 255] = 0
        seg_entropy = entropy(seg_logit_softmax)

        self.update_m(seg_label) 
        num_weight = self.get_num_weight(seg_label).to(torch.float32)
        entropy_weight = self.get_entropy_weight(seg_entropy, seg_label).to(torch.float32)

        weight =  entropy_weight * num_weight
        weight = weight / weight.mean()

        return weight

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