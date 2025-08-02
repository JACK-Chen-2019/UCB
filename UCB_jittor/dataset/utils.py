import numpy as np
#===xzy modified
# import torch
import jittor as jt
from jittor.dataset import Dataset
#===

def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs


def filter_images(dataset, labels, labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    if 0 in labels:
        labels.remove(0)

    print(f"Filtering images...")
    if labels_old is None:
        labels_old = []
    labels_cum = labels + labels_old + [0, 255]

    if overlap:
        fil = lambda c: any(x in labels for x in cls)
    else:
        fil = lambda c: any(x in labels for x in cls) and all(x in labels_cum for x in c)

    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if fil(cls):
            idxs.append(i)
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    return idxs


# class Subset(torch.utils.data.Dataset):
class Subset(Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
    """
    #===xzy modified
    def __init__(self, dataset, indices, transform=None, masking=False, tmp_labels=None, masking_value=None, inverted_order=None):
        
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.masking = masking
        self.tmp_labels = tmp_labels if tmp_labels is not None else []
        self.masking_value = masking_value
        self.inverted_order = inverted_order if inverted_order is not None else {}
    def target_transform(self, t):
        ori_shape = t.shape
        t = t.flatten().numpy()
        
        # t = np.apply_along_axis(lambda x: self.inverted_order[x] if x in self.tmp_labels else self.masking_value, 0, t)
        for i in range(len(t)):
            if t[i] in self.tmp_labels and t[i] in self.inverted_order:
                t[i] = self.inverted_order[t[i]]
            else:
                t[i] = self.masking_value

        t = jt.array(t)
        t = t.reshape(ori_shape)

        return t
    #===
    def __getitem__(self, idx):
        
        try:
            sample, target = self.dataset[self.indices[idx]]
        except Exception as e:
            raise Exception(
                f"dataset = {len(self.dataset)}, indices = {len(self.indices)}, idx = {idx}, msg = {str(e)}"
            )
        
        if self.transform is not None:

            sample, target = self.transform(sample, target)

        #===xzy modified
        if target.dtype == np.float32:
            target = target.astype(np.int8)
            target = jt.array(target)
        
        if self.masking is not None:
            target = self.target_transform(target)
        #===
        return sample, target

    def viz_getter(self, idx):
        image_path, raw_image, sample, target = self.dataset.viz_getter(self.indices[idx])
        if self.transform is not None:
            sample, target = self.transform(sample, target)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image_path, raw_image, sample, target

    def __len__(self):
        return len(self.indices)


class MaskLabels:
    """
    Use this class to mask labels that you don't want in your dataset.
    Arguments:
    labels_to_keep (list): The list of labels to keep in the target images
    mask_value (int): The value to replace ignored values (def: 0)
    """

    def __init__(self, labels_to_keep, mask_value=0):
        self.labels = labels_to_keep

        #===xzy modified 
        # self.value = torch.tensor(mask_value, dtype=torch.uint8)
        self.value = jt.Var(mask_value, dtype=jt.uint8)
        raise NotImplementedError("MaskLabels is not implemented in Jittor yet.")
        #===
    def __call__(self, sample):
        # sample must be a tensor

        #===xzy modified
        # assert isinstance(sample, torch.Tensor), "Sample must be a tensor"
        assert isinstance(sample, jt.Var), "Sample must be a Var"
        #===
        sample.apply_(lambda t: t.apply_(lambda x: x if x in self.labels else self.value))

        return sample
