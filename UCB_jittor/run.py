import copy
import os
import random

import numpy as np


#===xzy
# import torch
# from apex import amp
# from apex.parallel import DistributedDataParallel

# from torch import distributed
# from torch.utils import data
# from torch.utils.data.distributed import DistributedSampler


import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
from jittor.dataset import DataLoader
#===
import argparser
import tasks
import utils
#===xzy modified
# from dataset import (AdeSegmentationIncremental,
#                      CityscapesSegmentationIncrementalDomain,
#                      VOCSegmentationIncremental, transform)
from dataset import (VOCSegmentationIncremental, transform)

#===
from metrics import StreamSegMetrics
from segmentation_module import make_model
from train import Trainer
from utils.logger import Logger


def save_ckpt(path, model, trainer, optimizer, scheduler, epoch, best_score):
    # """ save current model
    # """
    # state = {
    #     "epoch": epoch,
    #     "model_state": model.state_dict(),
    #     "optimizer_state": optimizer.state_dict(),
    #     #===xzy modified
    #     # "scheduler_state": scheduler.state_dict(),
    #     #===
    #     "best_score": best_score,
    #     "trainer_state": trainer.state_dict()
    # }
    # # save model, using jittor's save_parameters
    
    # # torch.save(state, path) 
    # jt.save(state, path)

    model.save(path)


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose(
        [
            transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if opts.crop_val:
        val_transform = transform.Compose(
            [
                transform.Resize(size=opts.crop_size),
                transform.CenterCrop(size=opts.crop_size),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose(
            [
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cum = labels_old + labels

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(
        root=opts.data_root,
        train=True,
        transform=train_transform,
        labels=list(labels),
        labels_old=list(labels_old),
        idxs_path=path_base + f"/train-{opts.step}.npy",
        masking=not opts.no_mask,
        overlap=opts.overlap,
        disable_background=opts.disable_background,
        data_masking=opts.data_masking,
        test_on_val=opts.test_on_val,
        step=opts.step
    )
    ### Fix for classifier warm up
    tune_cls_dst = dataset(
        root=opts.data_root,
        train=True,
        transform=train_transform,
        labels=list(labels),
        labels_old=list(labels_old),
        idxs_path=path_base + f"/train-{opts.step}.npy",
        masking=not opts.no_mask,
        overlap=opts.overlap,
        disable_background=opts.disable_background,
        data_masking=opts.data_masking,
        test_on_val=opts.test_on_val,
        step=opts.step
    )
    if not opts.no_cross_val:  # if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst) - train_len
        train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    else:  # don't use cross_val
        val_dst = dataset(
            root=opts.data_root,
            train=False,
            transform=val_transform,
            labels=list(labels),
            labels_old=list(labels_old),
            idxs_path=path_base + f"/val-{opts.step}.npy",
            masking=not opts.no_mask,
            overlap=True,
            disable_background=opts.disable_background,
            data_masking=opts.data_masking,
            step=opts.step
        )

    image_set = 'train' if opts.val_on_trainset else 'val'
    test_dst = dataset(
        root=opts.data_root,
        train=opts.val_on_trainset,
        transform=val_transform,
        labels=list(labels_cum),
        idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy",
        disable_background=opts.disable_background,
        test_on_val=opts.test_on_val,
        step=opts.step,
        ignore_test_bg=opts.ignore_test_bg
    )
    
    return train_dst, tune_cls_dst, val_dst, test_dst, len(labels_cum)


def main(opts):
    #===xzy modified
    # distributed.init_process_group(backend='nccl', init_method='env://')
    # device_id, device = opts.local_rank, torch.device(opts.local_rank)
    # rank, world_size = distributed.get_rank(), distributed.get_world_size()
    # torch.cuda.set_device(device_id) 

    device = "cuda"

    jt.flags.use_cuda = 1
    #===

    

    if len(opts.lr) == 1 and len(opts.step) > 1:
        opts.lr = opts.lr * len(opts.step)

    os.makedirs("results", exist_ok=True)

    print(f"Learning for {len(opts.step)} with lrs={opts.lr}.")
    all_val_scores = []
    for i, (step, lr) in enumerate(zip(copy.deepcopy(opts.step), copy.deepcopy(opts.lr))):
        if i > 0:
            opts.step_ckpt = None

        opts.step = step
        opts.lr = lr

        val_score = run_step(opts, device)


    
        # if rank == 0:
        #     all_val_scores.append(val_score)

        # torch.cuda.empty_cache()

        # if rank == 0:
        #     with open(f"results/{opts.date}_{opts.dataset}_{opts.task}_{opts.name}.csv", "a+") as f:
        #         classes_iou = ','.join(
        #             [str(val_score['Class IoU'].get(c, 'x')) for c in range(opts.num_classes)]
        #         )
        #         f.write(f"{step},{classes_iou},{val_score['Mean IoU']}\n")

        all_val_scores.append(val_score)
     
        with open(f"results/{opts.date}_{opts.dataset}_{opts.task}_{opts.name}.csv", "a+") as f:
            classes_iou = ','.join(
                [str(val_score['Class IoU'].get(c, 'x')) for c in range(opts.num_classes)]
            )
            f.write(f"{step},{classes_iou},{val_score['Mean IoU']}\n")


# def run_step(opts, world_size, rank, device):

def run_step(opts, device):
    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    logger = Logger(logdir_full, rank=0, summary=False)

  

    # Set up random seed

    #===xzy modified
    # torch.manual_seed(opts.random_seed)
    # torch.cuda.manual_seed(opts.random_seed)
    jt.seed(opts.random_seed)
    jt.set_global_seed(opts.random_seed) 
    #===


    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    train_dst, tune_dst, val_dst, test_dst, n_classes = get_dataset(opts)





    # reset the seed, this revert changes in random seed
    
  
    #===xzy modified
    train_loader = DataLoader(
        train_dst,
        batch_size=opts.batch_size,
        drop_last=True
    )
    ### Fix for warm up
    tune_loader = DataLoader(
        tune_dst,
        batch_size=opts.batch_size,
        drop_last=True
    )
    tune_loader_select = DataLoader(
        tune_dst,
        batch_size=1,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dst,
        batch_size=opts.batch_size if opts.crop_val else 1,
    )
    #===
    

    opts.inital_nb_classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)[0]

    step_checkpoint = None
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))

   
    model = model.cuda()

    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")

    if opts.step == 0:  # if step 0, we don't need to instance the model_old
        model_old = None
    else:  # instance model_old
        model_old = make_model(
            opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1)
        )
        model_old = model_old.cuda()

    if opts.fix_bn:
        model.fix_bn()

    logger.debug(model)

    # xxx Set up optimizer
    params = []

    params_warm = []
    if not opts.freeze:
        if opts.step > 0:
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.body.parameters()),
                    'weight_decay': opts.weight_decay
                }
            )
        else:
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.body.parameters()),
                    'weight_decay': opts.weight_decay
                }
            )

    params.append(
        {
            "params": filter(lambda p: p.requires_grad, model.head.parameters()),
            'weight_decay': opts.weight_decay
        }
    )

    if opts.lr_old is not None and opts.step > 0:
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.cls[:-1].parameters()),
                'weight_decay': opts.weight_decay,
                "lr": opts.lr_old * opts.lr
            }
        )
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.cls[-1:].parameters()),
                'weight_decay': opts.weight_decay
            }
        )
    else:
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                'weight_decay': opts.weight_decay
            }
        )
      
        if opts.step > 0 and opts.warm_up:
            if opts.two_stage:
                params_warm.append(
                    {
                        "params": model.new_classifier_weight,
                        "weight_decay": opts.weight_decay,
                    }
                )
                params_warm.append(
                    {
                        "params": model.new_classifier_bias,
                        "weight_decay": opts.weight_decay,
                    }
                )
            else:
                params_warm.append(
                    {
                        "params": model.weight_old,
                        "weight_decay": opts.weight_decay,
                    }
                )
                params_warm.append(
                    {
                        "params": model.weight_new,
                        "weight_decay": opts.weight_decay,
                    }
                )
                params_warm.append(
                    {
                        "params": model.new_bias,
                        "weight_decay": opts.weight_decay,
                    }
                )
                params_warm.append(
                    {
                        "params": model.weight_old_bg,
                        "weight_decay": opts.weight_decay,
                    }
                )
                params_warm.append(
                    {
                        "params": model.weight_new_bg,
                        "weight_decay": opts.weight_decay,
                    }
                )


        else: # step == 0 or not warm 确保params_warm不为空
            params_warm.append(
                {
                "params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                'weight_decay': opts.weight_decay
                }
            )
 
    if model.scalar is not None:
        params.append({"params": model.scalar, 'weight_decay': opts.weight_decay})

    # optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    # tune_optimizer = torch.optim.SGD(params_warm, lr=opts.lr * opts.warm_lr_scale, momentum=0.9, nesterov=True) 
    optimizer = jt.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    tune_optimizer = jt.optim.SGD(params_warm, lr=opts.lr * opts.warm_lr_scale, momentum=0.9, nesterov=True) 



    if opts.lr_policy == 'poly':
        #===xzy modified
        # scheduler = utils.PolyLR(
        #     optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power
        # )
        # tune_scheduler = utils.PolyLR(
        #     tune_optimizer, max_iters=opts.warm_epochs * len(tune_loader), power=opts.lr_power
        # )
        scheduler = utils.PolyLR(
            optimizer, step_size=opts.epochs * len(train_loader)
        )
        tune_scheduler = utils.PolyLR(
            tune_optimizer, step_size=opts.warm_epochs * len(tune_loader)
        )
        #===
    else:
        raise NotImplementedError
    logger.debug("Optimizer:\n%s" % optimizer)

    if model_old is not None:
        
        # model_old.to(device)
        model_old = model_old.cuda()
 
        if opts.two_stage:
            model_old.new_classifier_weight = None 
            model_old.new_classifier_bias = None
        else:
            model_old.weight_new = None
            model_old.weight_old = None

            model_old.new_bias = None

            model_old.weight_new_bg = None
            model_old.weight_old_bg = None


    #===xzy modified
    #     if opts.warm_up:
    #         print("warm up!")
            
    #         [model, model_old], [optimizer, tune_optimizer] = amp.initialize(
    #             [model.to(device), model_old.to(device)], [optimizer, tune_optimizer], opt_level=opts.opt_level
    #         )
    #     else:
    #         [model, model_old], optimizer = amp.initialize(
    #                 [model.to(device), model_old.to(device)], optimizer, opt_level=opts.opt_level
    #             )
    #     model_old = DistributedDataParallel(model_old)
    # else:

    #     model, optimizer = amp.initialize(model.to(device), optimizer, opt_level=opts.opt_level)
    
    # model = DistributedDataParallel(model, delay_allreduce=True)
    #===

   

    # xxx Load old model from old weights if step > 0!
    if opts.step > 0:
        # get model path
        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            path = f"{opts.checkpoint}/{task_name}_{opts.name}_{opts.step - 1}.pkl"

        # generate model from path
        if os.path.exists(path):

            #===xzy modified
            # step_checkpoint = jt.load(path, map_location="cpu")
            # model.load_state_dict(
            #     step_checkpoint['model_state'], strict=False
            # )  # False because of incr. classifiers
            
            model.load(path)
            #==

            if opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                if opts.warm_up:
                    model.init_new_classifier_simplified(device)
                else:
                    model.init_new_classifier(device)
            elif opts.init_multimodal is not None:
                assert 1==2
                # model.init_new_classifier_multimodal(
                #     device, train_loader, opts.init_multimodal
                # )
            # Load state dict from the model state dict, that contains the old model parameters



            # model_old.load_state_dict(
            #     step_checkpoint['model_state'], strict=opts.strict_weights
            # )  # Load also here old parameters
            # logger.info(f"[!] Previous model loaded from {path}")
            # # clean memory
            # del step_checkpoint['model_state']

            model_old.load(path)
        elif opts.debug:
            logger.info(
                f"[!] WARNING: Unable to find of step {opts.step - 1}! Do you really want to do from scratch?"
            )
        else:
            raise FileNotFoundError(path)
        # put the old model into distributed memory and freeze it
        for par in model_old.parameters():
            par.requires_grad = False
        model_old.eval()

    # xxx Set up Trainer
    trainer_state = None
    # if not first step, then instance trainer from step_checkpoint
    if opts.step > 0 and step_checkpoint is not None:
        if 'trainer_state' in step_checkpoint:
            trainer_state = step_checkpoint['trainer_state']

    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(
        model,
        model_old,
        device=device,
        opts=opts,
        trainer_state=trainer_state,
        classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step),
        step=opts.step
    )

    # xxx Handle checkpoint for current model (model old will always be as previous step or None)
    best_score = 0.0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=opts.strict_weights)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        if 'trainer_state' in checkpoint:
            trainer.load_state_dict(checkpoint['trainer_state'])
        del checkpoint
    else:
        if opts.step == 0:
            logger.info("[!] Train from scratch")

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    #===xzy modified
    # if rank == 0 and opts.sample_num > 0:
    if opts.sample_num > 0:
    #===
        sample_ids = np.random.choice(
            len(val_loader), opts.sample_num, replace=False
        )  # sample idxs for visualization
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )  # de-normalization for original images

    TRAIN = not opts.test
    if opts.dataset == "cityscapes_domain":
        val_metrics = StreamSegMetrics(opts.num_classes)
    else:
        val_metrics = StreamSegMetrics(n_classes)
    results = {}

   

    if TRAIN:
        
        classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
        
        if opts.warm_up is True and len(classes) > 1:
            bucket = trainer.select(
                train_loader=tune_loader_select,
                logger=logger
            )
            

            model.reset_weight_old(bucket)
            for cur_epoch in range(opts.warm_epochs):
                model.train()
                
                warm_epoch_loss = trainer.warm_up(
                    cur_epoch=cur_epoch,
                    optim=tune_optimizer,
                    train_loader=tune_loader,
                    scheduler=tune_scheduler,
                    logger=logger
                )

                logger.info(
                f"End of Warm Epoch {cur_epoch}/{opts.warm_epochs},"
                f"Class loss={warm_epoch_loss}"
                )
            #===xzy modified
            # model.init_via_weight()
            model.init_via_weight()
            #===
 
        del tune_optimizer, tune_scheduler
        #===xzy modified
        # if opts.two_stage:
        #     model.new_classifier_weight = None 
        #     model.new_classifier_bias = None 
        # else:
        #     model.weight_new = None
        #     model.weight_old = None

        #     model.new_bias = None

        #     model.weight_new_bg = None
        #     model.weight_old_bg = None
        if opts.two_stage:
            model.new_classifier_weight = None 
            model.new_classifier_bias = None 
        else:
            model.weight_new = None
            model.weight_old = None

            model.new_bias = None

            model.weight_new_bg = None
            model.weight_old_bg = None
        #===

        if opts.warm_up:
            for param in model.parameters():
                param.requires_grad = True
        for cur_epoch in range(opts.epochs):
            # =====  Train  =====
            
            model.train()

            epoch_loss = trainer.train(
                cur_epoch=cur_epoch,
                optim=optimizer,
                train_loader=train_loader,
                scheduler=scheduler,
                logger=logger
            )

            logger.info(
                f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0]+epoch_loss[1]},"
                f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}"
            )

            # # =====  Log metrics on Tensorboard =====
            # logger.add_scalar("E-Loss", epoch_loss[0] + epoch_loss[1], cur_epoch)
            # logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
            # logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)

            # =====  Validation  =====
            if (cur_epoch + 1) % opts.val_interval == 0:
                logger.info("validate on val set...")
                model.eval()
                val_loss, val_score, ret_samples = trainer.validate(
                    loader=val_loader,
                    metrics=val_metrics,
                    ret_samples_ids=sample_ids,
                    logger=logger
                )

                # logger.print("Done validation")
                # logger.info(
                #     f"End of Validation {cur_epoch}/{opts.epochs}, Validation Loss={val_loss[0]+val_loss[1]},"
                #     f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}"
                # )

                # logger.info(val_metrics.to_str(val_score))

                # =====  Save Best Model  =====
                if True:#rank == 0:  # save best model at the last iteration
                    score = val_score['Mean IoU']
                    # best model to build incremental steps
                    save_ckpt(
                        f"{opts.checkpoint}/{task_name}_{opts.name}_{opts.step}.pkl", model,
                        trainer, optimizer, scheduler, cur_epoch, score
                    )
                    # logger.info("[!] Checkpoint saved.")

                # =====  Log metrics on Tensorboard =====
                # visualize validation score and samples
                # logger.add_scalar("V-Loss", val_loss[0] + val_loss[1], cur_epoch)
                # logger.add_scalar("V-Loss-reg", val_loss[1], cur_epoch)
                # logger.add_scalar("V-Loss-cls", val_loss[0], cur_epoch)
                # logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
                # logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
                # logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
                # logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
                # logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)

                # keep the metric to print them at the end of training
                results["V-IoU"] = val_score['Class IoU']
                results["V-Acc"] = val_score['Class Acc']

                for k, (img, target, lbl) in enumerate(ret_samples):
                    img = (denorm(img) * 255).astype(np.uint8)
                    target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
                    lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

                    concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                    logger.add_image(f'Sample_{k}', concat_img, cur_epoch)



    del tune_loader, tune_loader_select
    # =====  Save Best Model at the end of training =====

    # if rank == 0 and TRAIN:  # save best model at the last iteration
    if TRAIN:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(
            f"{opts.checkpoint}/{task_name}_{opts.name}_{opts.step}.pkl", model, trainer, optimizer,
            scheduler, cur_epoch, best_score
        )
        logger.info("[!] Checkpoint saved.")

    # torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    # test_loader = data.DataLoader(
    #     test_dst,
    #     batch_size=opts.batch_size if opts.crop_val else 1,
    #     sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
    #     num_workers=opts.num_workers
    # )
    test_loader = DataLoader(
        test_dst,
        batch_size=opts.batch_size if opts.crop_val else 1,
    )

    # load best model
    if True: #TRAIN:
        # Always reloading model for now
        # https://github.com/arthurdouillard/CVPR2021_PLOP/issues/3
        model = make_model(
            opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
        )
        # Put the model on GPU

        # apex
        if opts.two_stage:
            model.new_classifier_weight = None 
            model.new_classifier_bias = None 
        else:
            model.weight_new = None
            model.weight_old = None
            model.new_bias = None

            model.weight_new_bg = None
            model.weight_old_bg = None


        # model = DistributedDataParallel(model.cuda(device))
        


        # ckpt = f"{opts.checkpoint}/{task_name}_{opts.name}_{opts.step}.pkl"
        # checkpoint = torch.load(ckpt, map_location="cpu")
        # model.load_state_dict(checkpoint["model_state"])
        # logger.info(f"*** Model restored from {ckpt}")
        # del checkpoint
        # trainer = Trainer(model, None, device=device, opts=opts, step=opts.step)

        ckpt = f"{opts.checkpoint}/{task_name}_{opts.name}_{opts.step}.pkl"
        checkpoint = model.load(ckpt)
       
        del checkpoint
        trainer = Trainer(model, None, device=device, opts=opts, step=opts.step)

    model.eval()

    val_loss, val_score, _ = trainer.validate(
        loader=test_loader, metrics=val_metrics, logger=logger, end_task=True
    )
    # logger.print("Done test")
    # logger.info(
    #     f"*** End of Test, Total Loss={val_loss[0]+val_loss[1]},"
    #     f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}"
    # )
    # logger.info(val_metrics.to_str(val_score))
    # logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    # logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    # logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    # results["T-IoU"] = val_score['Class IoU']
    # results["T-Acc"] = val_score['Class Acc']
    # logger.add_results(results)

    # logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    # logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    # logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)

    logger.close()

    del model
    if model_old is not None:
        del model_old

    return val_score


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs(f"{opts.checkpoint}", exist_ok=True)

    main(opts)
