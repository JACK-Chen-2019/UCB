import argparse
import json

import tasks


def modify_command_options(opts):
    if opts.dataset == 'voc':
        opts.num_classes = 21
    elif opts.dataset == 'ade':
        opts.num_classes = 151
    elif opts.dataset == "cityscapes":
        opts.num_classes = 17
    elif opts.dataset == "cityscapes_domain":
        opts.num_classes = 19
    elif opts.dataset == "cityscapes_classdomain":
        opts.num_classes = 19
    else:
        raise NotImplementedError(f"Unknown dataset: {opts.dataset}")

    if not opts.visualize:
        opts.sample_num = 0

    if opts.method is not None:
        if opts.method == 'FT':
            pass
        if opts.method == 'LWF':
            opts.loss_kd = 100
        if opts.method == 'LWF-MC':
            opts.icarl = True
            opts.icarl_importance = 10
        if opts.method == 'ILT':
            opts.loss_kd = 100
            opts.loss_de = 100
        if opts.method == 'EWC':
            opts.regularizer = "ewc"
            opts.reg_importance = 100
        if opts.method == 'RW':
            opts.regularizer = "rw"
            opts.reg_importance = 100
        if opts.method == 'PI':
            opts.regularizer = "pi"
            opts.reg_importance = 100
        if opts.method == 'MiB':
            opts.loss_kd = 10
            opts.unce = True
            opts.unkd = True
            opts.init_balanced = True
            opts.pseudo = "naive"
        if opts.method == 'PLOP':
            opts.pod = "local"
            opts.pod_factor = 0.01
            opts.pod_logits = True
            opts.pod_options = {"switch": {"after": {"extra_channels": "sum", "factor": 0.0005, "type": "local"}}}
            opts.pseudo = "entropy"
            opts.threshold = 0.001
            opts.classif_adaptive_factor = True
            opts.init_balanced = True
        if opts.method == 'PLOP_Entropy':
            opts.pod = "local"
            opts.pod_factor = 0.01
            opts.pod_logits = True
            opts.pod_options = {"switch": {"after": {"extra_channels": "sum", "factor": 0.0005, "type": "local"}}}
            opts.pseudo = "naive"
            opts.threshold = 0.001
            opts.classif_adaptive_factor = True
            opts.init_balanced = True


    opts.no_overlap = not opts.overlap
    opts.no_cross_val = not opts.cross_val

    return opts


def get_argparser():
    parser = argparse.ArgumentParser()

    # Performance Options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=1, help='number of workers (default: 1)')

    # Datset Options
    parser.add_argument("--data_root", type=str, default='data', help="path to Dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default='voc',
        choices=['voc', 'ade', 'cityscapes_domain'],
        help='Name of dataset'
    )
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")
    parser.add_argument(
        "--dont_predict_bg", action="store_true", default=False, help="Useful for cityscapes"
    )

    # Method Options
    # BE CAREFUL USING THIS, THEY WILL OVERRIDE ALL THE OTHER PARAMETERS.
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=['FT', 'LWF', 'LWF-MC', 'ILT', 'EWC', 'RW', 'PI', 'MiB', 'PLOP', 'PLOP_Entropy'],
        help="The method you want to use. BE CAREFUL USING THIS, IT MAY OVERRIDE OTHER PARAMETERS."
    )

    parser.add_argument("--strict_weights", action="store_false", default=True)
    parser.add_argument("--base_weights", action="store_true", default=False)

    # Train Options
    parser.add_argument("--epochs", type=int, default=30, help="epoch number (default: 30)")
    parser.add_argument(
        "--fix_bn",
        action='store_true',
        default=False,
        help='fix batch normalization during training (default: False)'
    )

    parser.add_argument("--batch_size", type=int, default=4, help='batch size (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512, help="crop size (default: 513)")

    parser.add_argument(
        "--lr", type=float, nargs="+", default=[0.007], help="learning rate (default: 0.007)"
    )
    parser.add_argument(
        "--lr_old", type=float, default=None, help="learning rate for old classes weights."
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help='momentum for SGD (default: 0.9)'
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)'
    )

    parser.add_argument(
        "--lr_policy",
        type=str,
        default='poly',
        choices=['poly', 'step'],
        help="lr schedule policy (default: poly)"
    )
    parser.add_argument(
        "--lr_decay_step", type=int, default=5000, help="decay step for stepLR (default: 5000)"
    )
    parser.add_argument(
        "--lr_decay_factor", type=float, default=0.1, help="decay factor for stepLR (default: 0.1)"
    )
    parser.add_argument(
        "--lr_power", type=float, default=0.9, help="power for polyLR (default: 0.9)"
    )
    parser.add_argument(
        "--bce", default=False, action='store_true', help="Whether to use BCE or not (default: no)"
    )

    # Validation Options
    parser.add_argument("--test_on_val", default=False, action="store_true")
    parser.add_argument(
        "--val_on_trainset",
        action='store_true',
        default=False,
        help="enable validation on train set (default: False)"
    )
    parser.add_argument(
        "--cross_val",
        action='store_true',
        default=False,
        help="If validate on training or on validation (default: Train)"
    )
    parser.add_argument(
        "--crop_val",
        action='store_false',
        default=True,
        help='do crop for validation (default: True)'
    )

    # Logging Options
    parser.add_argument(
        "--logdir", type=str, default='./logs', help="path to Log directory (default: ./logs)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default='Experiment',
        help="name of the experiment - to append to log directory (default: Experiment)"
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=30,
        help='number of samples for visualization (default: 0)'
    )
    parser.add_argument("--debug", action='store_true', default=False, help="verbose option")
    parser.add_argument(
        "--visualize",
        action='store_false',
        default=True,
        help="visualization on tensorboard (def: Yes)"
    )
    parser.add_argument(
        "--print_interval", type=int, default=10, help="print interval of loss (default: 10)"
    )
    parser.add_argument(
        "--val_interval", type=int, default=30, help="epoch interval for eval (default: 1)"
    )
    parser.add_argument(
        "--ckpt_interval", type=int, default=1, help="epoch interval for saving model (default: 1)"
    )

    # Model Options
    parser.add_argument(
        "--backbone",
        type=str,
        default='resnet101',
        choices=['resnet50', 'resnet101'],
        help='backbone for the body (def: resnet50)'
    )
    parser.add_argument(
        "--output_stride",
        type=int,
        default=16,
        choices=[8, 16],
        help='stride for the backbone (def: 16)'
    )
    parser.add_argument(
        "--no_pretrained",
        action='store_true',
        default=False,
        help='Wheather to use pretrained or not (def: True)'
    )
    parser.add_argument(
        "--norm_act",
        type=str,
        default="iabn_sync",
        choices=['iabn_sync', 'iabn', 'abn', 'std', 'iabn_sync_test'],
        help='Which BN to use (def: abn_sync'
    )

    parser.add_argument(
        "--fusion-mode",
        metavar="NAME",
        type=str,
        choices=["mean", "voting", "max"],
        default="mean",
        help="How to fuse the outputs. Options: 'mean', 'voting', 'max'"
    )
    parser.add_argument(
        "--pooling",
        type=int,
        default=32,
        help='pooling in ASPP for the validation phase (def: 32)'
    )

    # Test and Checkpoint options
    parser.add_argument(
        "--test",
        action='store_true',
        default=False,
        help="Whether to train or test only (def: train and test)"
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        type=str,
        help="path to trained model. Leave it None if you want to retrain your model"
    )

    # Parameters for Knowledge Distillation of ILTSS (https://arxiv.org/abs/1907.13372)
    parser.add_argument(
        "--freeze",
        action='store_true',
        default=False,
        help="Use this to freeze the feature extractor in incremental steps"
    )
    parser.add_argument(
        "--loss_de",
        type=float,
        default=0.,  # Distillation on Encoder
        help="Set this hyperparameter to a value greater than "
        "0 to enable distillation on Encoder (L2)"
    )
    parser.add_argument(
        "--loss_kd",
        type=float,
        default=0.,  # Distillation on Output
        help="Set this hyperparameter to a value greater than "
        "0 to enable Knowlesge Distillation (Soft-CrossEntropy)"
    )

    # Parameters for EWC, RW, and SI (from Riemannian Walks https://arxiv.org/abs/1801.10112)
    parser.add_argument(
        "--regularizer",
        default=None,
        type=str,
        choices=['ewc', 'rw', 'pi'],
        help="regularizer you want to use. Default is None"
    )
    parser.add_argument(
        "--reg_importance",
        type=float,
        default=1.,
        help="set this par to a value greater than 0 to enable regularization"
    )
    parser.add_argument(
        "--reg_alpha",
        type=float,
        default=0.9,
        help="Hyperparameter for RW and EWC that controls the update of Fisher Matrix"
    )
    parser.add_argument(
        "--reg_no_normalize",
        action='store_true',
        default=False,
        help="If EWC, RW, PI must be normalized or not"
    )
    parser.add_argument(
        "--reg_iterations",
        type=int,
        default=10,
        help="If RW, the number of iterations after each the update of the score is done"
    )

    # Arguments for ICaRL (from https://arxiv.org/abs/1611.07725)
    parser.add_argument(
        "--icarl", default=False, action='store_true', help="If enable ICaRL or not (def is not)"
    )
    parser.add_argument(
        "--icarl_importance",
        type=float,
        default=1.,
        help="the regularization importance in ICaRL (def is 1.)"
    )
    parser.add_argument(
        "--icarl_disjoint",
        action='store_true',
        default=False,
        help="Which version of icarl is to use (def: combined)"
    )
    parser.add_argument(
        "--icarl_bkg",
        action='store_true',
        default=False,
        help="If use background from GT (def: No)"
    )

    # METHODS
    parser.add_argument(
        "--init_balanced",
        default=False,
        action='store_true',
        help="Enable Background-based initialization for new classes"
    )
    parser.add_argument(
        "--unkd",
        default=False,
        action='store_true',
        help="Enable Unbiased Knowledge Distillation instead of Knowledge Distillation"
    )
    parser.add_argument(
        "--alpha",
        default=1.,
        type=float,
        help="The parameter to hard-ify the soft-labels. Def is 1."
    )
    parser.add_argument(
        "--unce",
        default=False,
        action='store_true',
        help="Enable Unbiased Cross Entropy instead of CrossEntropy"
    )

    # Incremental parameters
    parser.add_argument(
        "--task",
        type=str,
        default="19-1",
        choices=tasks.get_task_list(),
        help="Task to be executed (default: 19-1)"
    )
    parser.add_argument(
        "--step",
        type=int,
        nargs="+",
        default=[0],
        help="The incremental step in execution (default: 0)"
    )
    parser.add_argument(
        "--no_mask",
        action='store_true',
        default=False,
        help="Use this to not mask the old classes in new training set"
    )
    parser.add_argument(
        "--data_masking",
        type=str,
        default="current",
        choices=["current", "current+old", "all", "new"]
    )

    parser.add_argument(
        "--overlap",
        action='store_true',
        default=False,
        help="Use this to not use the new classes in the old training set"
    )
    parser.add_argument(
        "--step_ckpt",
        default=None,
        type=str,
        help="path to trained model at previous step. Leave it None if you want to use def path"
    )
    parser.add_argument('--opt_level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O0')

    # Pseudo-labeling
    parser.add_argument(
        "--pseudo",
        type=str,
        default=None,
        help="Pseudo-labeling method." +
        ", ".join(["naive", "confidence", "threshold_5", "threshold_8", "median", "entropy"])
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--step_threshold", type=float, default=None)
    parser.add_argument(
        "--ce_on_pseudo",
        default=False,
        action="store_true",
        help="Pseudo Labels are trained w/ CE, default criterion for others"
    )
    parser.add_argument("--pseudo_nb_bins", default=None, type=int)
    parser.add_argument("--classif_adaptive_factor", default=False, action="store_true")
    parser.add_argument("--classif_adaptive_min_factor", default=0.0, type=float)
    parser.add_argument("--pseudo_soft", default=None, type=str, choices=["soft_certain", "soft_uncertain"])
    parser.add_argument("--pseudo_soft_factor", default=1.0, type=float)
    parser.add_argument("--pseudo_ablation", default=None, choices=["corrected_errors", "removed_errors"])


    parser.add_argument("--kd_new", default=False, action="store_true", help="Apply KD only on new")

    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/step"
    )

    parser.add_argument(
        "--pod",
        default=None,
        type=str,
        choices=[
            "spatial", "local", "global"
        ]
    )
    parser.add_argument("--pod_factor", default=5., type=float)
    parser.add_argument("--pod_options", default=None, type=json.loads)
    parser.add_argument("--pod_prepro", default="pow", type=str)
    parser.add_argument("--no_pod_schedule", default=False, action="store_true")
    parser.add_argument(
        "--pod_apply", default="all", type=str, choices=["all", "backbone", "deeplab"]
    )
    parser.add_argument("--pod_deeplab_mask", default=False, action="store_true")
    parser.add_argument(
        "--pod_deeplab_mask_factor", default=None, type=float, help="By default as the POD factor"
    )
    parser.add_argument("--deeplab_mask_downscale", action="store_true", default=False)
    parser.add_argument("--pod_interpolate_last", default=False, action="store_true")
    parser.add_argument(
        "--pod_logits", default=False, action="store_true", help="Also apply POD to logits."
    )
    parser.add_argument(
        "--pod_large_logits", default=False, action="store_true", help="Also apply POD to large logits."
    )
    parser.add_argument("--spp_scales", default=[1, 2, 4], type=int, nargs="+")

    parser.add_argument("--date", default="", type=str)

    parser.add_argument("--nb_background_modes", default=1, type=int)
    parser.add_argument(
        "--init_multimodal",
        default=None,
        type=str,
        choices=["max", "softmax", "max_init", "softmax_init", "softmax_remove", "max_remove"]
    )
    parser.add_argument("--multimodal_fusion", default="sum", type=str)

    parser.add_argument("--align_weight", default=None, choices=["old", "background", "all"])
    parser.add_argument(
        "--align_weight_frequency", default="never", choices=["never", "epoch", "task"]
    )

    parser.add_argument("--cosine", default=False, action="store_true")
    parser.add_argument("--nca", default=False, action="store_true")
    parser.add_argument("--nca_margin", default=0., type=float)

    parser.add_argument("--kd_mask", default=None, choices=["oldbackground", "new"])
    parser.add_argument("--kd_mask_adaptative_factor", default=False, action="store_true")

    parser.add_argument(
        "--disable_background",
        action="store_true",
        help="Remove the fake background, only for Cityscapes. DEPRECATED TO REMOVE"
    )
    parser.add_argument("--ignore_test_bg", action="store_true", default=False)

    parser.add_argument(
        "--entropy_min",
        default=0.,
        type=float,
        help="Factor for the entropy minimization (cf advent)"
    )
    parser.add_argument("--entropy_min_mean_pixels", default=False, action="store_true", help="")

    parser.add_argument("--kd_scheduling", default=False, action="store_true")

    parser.add_argument("--sample_weights_new", default=None, type=float)

    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--temperature_semiold", default=1.0, type=float)
    parser.add_argument("--temperature_apply", default=None, choices=["all", "new", "old"])

    parser.add_argument("--code_directory", default=".")

    parser.add_argument("--kd_bce_sig", action="store_true", default=False)
    parser.add_argument("--kd_bce_sig_shape", choices=["trim", "sum"], default="trim")

    parser.add_argument("--exkd_gt", action="store_true", default=False)
    parser.add_argument("--exkd_sum", action="store_true", default=False)


    parser.add_argument("--focal_loss", action="store_true", default=False)
    parser.add_argument("--focal_loss_new", action="store_true", default=False)
    parser.add_argument("--focal_loss_gamma", default=2, type=int)

    parser.add_argument("--ce_on_new", default=False, action="store_true")

    return parser
