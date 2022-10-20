import datetime
import os
import time

import numpy as np
import torch
import torch.optim as optim
import util.utils as utils
from checkpoint import align_and_update_state_dicts, checkpoint, strip_prefix_if_present
from criterion_fs import FSInstSetCriterion
from datasets.scannetv2_fs_inst_block import FSInstDataset
from model.geoformer.geoformer_fs import GeoFormerFS

from tensorboardX import SummaryWriter
from util.config import cfg
from util.dataloader_util import get_rank
from util.log import create_logger
from util.utils_scheduler import adjust_learning_rate

torch.autograd.set_detect_anomaly(True)

def init():
    os.makedirs(cfg.exp_path, exist_ok=True)
    # log the config
    global logger
    logger = create_logger()
    logger.info(cfg)
    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)


def train_one_epoch(start_epoch, train_loader, model, criterion, optimizer, cfg):

    data_time = utils.AverageMeter()

    model.train()
    net_device = next(model.parameters()).device

    check_time = time.time()

    for iteration, batch in enumerate(train_loader):
        data_time.update(time.time() - check_time)
        torch.cuda.empty_cache()
        support_dict, query_dict, scene_infos = batch

        for key in support_dict:
            if torch.is_tensor(support_dict[key]):
                support_dict[key] = support_dict[key].to(net_device)
        for key in query_dict:
            if torch.is_tensor(query_dict[key]):
                query_dict[key] = query_dict[key].to(net_device)

        corr_save_path = os.path.join(cfg.output_path, "corr_{}".format(start_epoch), "iter_{}".format(iteration))
        os.makedirs(corr_save_path, exist_ok=True)

        model(support_dict, query_dict, remember=False, training=True,
              show_corr=True, corr_save_path=corr_save_path)


if __name__ == "__main__":
    # init
    init()

    torch.cuda.set_device(0)
    np.random.seed(cfg.manual_seed + get_rank())
    torch.manual_seed(cfg.manual_seed + get_rank())
    torch.cuda.manual_seed_all(cfg.manual_seed + get_rank())

    # model
    logger.info("=> creating model ...")

    model = GeoFormerFS()
    model = model.cuda()

    logger.info("# training parameters: {}".format(sum([x.nelement() for x in model.parameters() if x.requires_grad])))

    criterion = FSInstSetCriterion()
    criterion = criterion.cuda()

    # optimizer
    if cfg.optim == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == "SGD":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    logger.info(f"Learning rate: {cfg.lr}")
    start_epoch = -1

    if cfg.pretrain:
        logger.info("=> loading checkpoint '{}'".format(cfg.pretrain))
        loaded = torch.load(cfg.pretrain, map_location=torch.device("cpu"))["state_dict"]
        model_state_dict = model.state_dict()
        loaded_state_dict = strip_prefix_if_present(loaded, prefix="module.")
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        model.load_state_dict(model_state_dict)
        logger.info("=> done loading pretrain")

    if cfg.resume:
        checkpoint_fn = cfg.resume
        if os.path.isfile(checkpoint_fn):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn, map_location=torch.device("cpu"))
            start_epoch = state["epoch"] + 1

            model_state_dict = model.state_dict()
            loaded_state_dict = strip_prefix_if_present(state["state_dict"], prefix="module.")
            align_and_update_state_dicts(model_state_dict, loaded_state_dict)
            model.load_state_dict(model_state_dict)

            logger.info("=> loaded checkpoint '{}' (start_epoch {})".format(checkpoint_fn, start_epoch))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

    dataset = FSInstDataset(split_set="train")
    train_loader = dataset.trainLoader()

    train_one_epoch(start_epoch, train_loader, model, criterion, optimizer, cfg)
