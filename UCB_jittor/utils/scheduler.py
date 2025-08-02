#===xzy modified
# from torch.optim.lr_scheduler import _LRScheduler, StepLR
# class PolyLR(_LRScheduler):
#     def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
#         self.power = power
#         self.max_iters = max_iters
#         super(PolyLR, self).__init__(optimizer, last_epoch)
    
#     def get_lr(self):
#         return [ base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power
#                 for base_lr in self.base_lrs]



from jittor.lr_scheduler import StepLR

class PolyLR(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        super(PolyLR, self).__init__(optimizer, step_size, gamma, last_epoch)
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.cur_epoch = 0
    
    def get_lr(self):
        # refer: https://github.com/Jittor/jittor/blob/master/python/jittor/lr_scheduler.py and https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py
        # return [ base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power
        #         for base_lr in self.base_lrs]
        raise NotImplementedError
    

#===