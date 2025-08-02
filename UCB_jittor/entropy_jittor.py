import jittor as jt
import jittor.nn as nn
import numpy as np
import math

jt.flags.use_cuda = 1  # 设置为 1 可启用 GPU

# ========== 定义 entropy 计算 ==========
def entropy(probabilities):
    c = probabilities.shape[1]
    factor = c / math.log(c + 1e-8)
    log_p = jt.log(probabilities + 1e-8)
    entropy_map = -factor * (probabilities * log_p).mean(dim=1)
    return entropy_map

# ========== 定义主类 ==========
class EntropyWeights:
    def __init__(self, tot_classes, old_classes):
        self.tot_classes = tot_classes
        self.old_classes = old_classes
        self.num = 0
        self.save_entropy = []

    def update_m(self, seg_label):
        seg_label = seg_label.stop_grad()
        seg_label = jt.where(seg_label == 255, jt.zeros_like(seg_label), seg_label)
        B = seg_label.shape[0]

        histogram = []
        for i in range(self.tot_classes):
            histogram.append((seg_label == i).sum())
        histogram = jt.stack(histogram).float()
        histogram = histogram / (histogram.sum() + 1e-8)

        if not hasattr(self, 'm'):
            self.m = histogram
        else:
            self.m = (self.m * self.num + histogram * B) / (self.num + B)
        self.num += B

    def get_num_weight(self, seg_label):
        seg_label = seg_label.stop_grad()
        num_weight = jt.zeros_like(seg_label).float()

        class_num = (self.m > 0).sum()

        # 正确计算 class_weight
        class_weight = self.m[1:].sum() / ((class_num - 1) * self.m + 1e-8)
        class_weight = class_weight.reshape((-1,))
        class_weight = jt.concat([jt.array([1.]), class_weight])  # 拼接时保持一致维度

        # 范围限制
        low = np.sqrt(float((self.tot_classes - self.old_classes) / self.old_classes))
        high = np.sqrt(float((self.old_classes) / (self.tot_classes - self.old_classes)))
        class_weight = jt.clamp(class_weight, min_v=low, max_v=high)

        # 根据标签值生成权重图
        for i in range(self.tot_classes):
            num_weight = jt.where(seg_label == i, class_weight[i], num_weight)

        return num_weight

    def get_entropy_weight(self, seg_entropy, seg_label):
        min_val = seg_entropy.min()
        max_val = seg_entropy.max()
        seg_entropy = (seg_entropy - min_val) / (max_val - min_val + 1e-8)

        entropy_high = 0.8
        mask_new = (seg_label >= self.old_classes)
        seg_entropy = jt.where(mask_new, jt.zeros_like(seg_entropy), seg_entropy)

        scale = 1 / (entropy_high + 1e-8)
        entropy_weight = jt.exp(-scale * (seg_entropy - entropy_high))
        entropy_weight = jt.where((seg_entropy >= entropy_high) & (seg_label < self.old_classes),
                                  jt.zeros_like(entropy_weight), entropy_weight)

        return entropy_weight

    def get_weight(self, seg_label, seg_output):
        seg_output = seg_output.float()
        seg_logit_softmax = nn.softmax(seg_output, dim=1)
        seg_label = jt.where(seg_label == 255, jt.zeros_like(seg_label), seg_label)

        seg_entropy = entropy(seg_logit_softmax)
        self.update_m(seg_label)

        num_weight = self.get_num_weight(seg_label).float()
        entropy_weight = self.get_entropy_weight(seg_entropy, seg_label).float()

        weight = entropy_weight * num_weight
        weight = weight / (weight.mean() + 1e-8)
        return weight


# ========== 测试入口 ==========
if __name__ == '__main__':
    B, C, H, W = 2, 6, 4, 4
    tot_classes = 6
    old_classes = 3

    seg_output = jt.randn(B, C, H, W)

    seg_label_np = np.random.randint(0, tot_classes, size=(B, H, W)).astype(np.int32)
    seg_label_np[0, 0, 0] = 255  # ignore 区
    seg_label = jt.array(seg_label_np)

    ew = EntropyWeights(tot_classes=tot_classes, old_classes=old_classes)
    weight_map = ew.get_weight(seg_label, seg_output)

    print("Seg label:\n", seg_label)
    print("Weight map shape:", weight_map.shape)
    print("Weight mean:", weight_map.mean())
    print("Weight min/max:", weight_map.min(), "/", weight_map.max())