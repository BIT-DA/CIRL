import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from models.model_factory import *
from optimizer.optimizer_helper import get_optim_and_scheduler
from data import *
from utils.Logger import Logger
from utils.tools import *
from models.classifier import Masker

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--input_dir", default=None, help="The directory of dataset lists")
    parser.add_argument("--output_dir", default=None, help="The directory to save logs and models")
    parser.add_argument("--config", default=None, help="Experiment configs")
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=True, help="If true will save tensorboard compatible logs")
    args = parser.parse_args()
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


class Trainer:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0

        # networks
        self.encoder = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.classifier = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)
        self.classifier_ad = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)
        dim = self.config["networks"]["classifier"]["in_dim"]
        self.masker = Masker(in_dim=dim, num_classes = dim,middle = 4*dim,k=self.config["k"]).to(device)
       
        # optimizers
        self.encoder_optim, self.encoder_sched = \
            get_optim_and_scheduler(self.encoder, self.config["optimizer"]["encoder_optimizer"])
        self.classifier_optim, self.classifier_sched = \
            get_optim_and_scheduler(self.classifier, self.config["optimizer"]["classifier_optimizer"])
        self.classifier_ad_optim, self.classifier_ad_sched = \
            get_optim_and_scheduler(self.classifier_ad, self.config["optimizer"]["classifier_optimizer"])
        self.masker_optim, self.masker_sched = \
            get_optim_and_scheduler(self.masker, self.config["optimizer"]["classifier_optimizer"])

        # dataloaders
        self.train_loader = get_fourier_train_dataloader(args=self.args, config=self.config)
        self.val_loader = get_val_dataloader(args=self.args, config=self.config)
        self.test_loader = get_test_loader(args=self.args, config=self.config)
        self.eval_loader = {'val': self.val_loader, 'test': self.test_loader}

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()

        # turn on train mode
        self.encoder.train()
        self.classifier.train()
        self.classifier_ad.train()
        self.masker.train()

        for it, (batch, label, domain) in enumerate(self.train_loader):

            # preprocessing
            batch = torch.cat(batch, dim=0).to(self.device)
            labels = torch.cat(label, dim=0).to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1
            # zero grad
            self.encoder_optim.zero_grad()
            self.classifier_optim.zero_grad()
            self.classifier_ad_optim.zero_grad()
            self.masker_optim.zero_grad()

            # forward
            loss_dict = {}
            correct_dict = {}
            num_samples_dict = {}
            total_loss = 0.0

            ## --------------------------step 1 : update G and C -----------------------------------
            features = self.encoder(batch)
            masks_sup = self.masker(features.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            if self.current_epoch <= 5:
                masks_sup = torch.ones_like(features.detach())
                masks_inf = torch.ones_like(features.detach())
            features_sup = features * masks_sup
            features_inf = features * masks_inf
            scores_sup = self.classifier(features_sup)
            scores_inf = self.classifier_ad(features_inf)

            assert batch.size(0) % 2 == 0
            split_idx = int(batch.size(0) / 2)
            features_ori, features_aug = torch.split(features, split_idx)
            assert features_ori.size(0) == features_aug.size(0)

            # classification loss for sup feature
            loss_cls_sup = criterion(scores_sup, labels)
            loss_dict["sup"] = loss_cls_sup.item()
            correct_dict["sup"] = calculate_correct(scores_sup, labels)
            num_samples_dict["sup"] = int(scores_sup.size(0))

            # classification loss for inf feature
            loss_cls_inf = criterion(scores_inf, labels)
            loss_dict["inf"] = loss_cls_inf.item()
            correct_dict["inf"] = calculate_correct(scores_inf, labels)
            num_samples_dict["inf"] = int(scores_inf.size(0))

            # factorization loss for features between ori and aug
            loss_fac = factorization_loss(features_ori,features_aug)
            loss_dict["fac"] = loss_fac.item()

            # get consistency weight
            const_weight = get_current_consistency_weight(epoch=self.current_epoch,
                                                          weight=self.config["lam_const"],
                                                          rampup_length=self.config["warmup_epoch"],
                                                          rampup_type=self.config["warmup_type"])

            # calculate total loss
            total_loss = 0.5*loss_cls_sup + 0.5*loss_cls_inf + const_weight*loss_fac
            loss_dict["total"] = total_loss.item()

            # backward
            total_loss.backward()

            # update
            self.encoder_optim.step()
            self.classifier_optim.step()
            self.classifier_ad_optim.step()


            ## ---------------------------------- step2: update masker------------------------------
            self.masker_optim.zero_grad()
            features = self.encoder(batch)
            masks_sup = self.masker(features.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            features_sup = features * masks_sup
            features_inf = features * masks_inf
            scores_sup = self.classifier(features_sup)
            scores_inf = self.classifier_ad(features_inf)

            loss_cls_sup = criterion(scores_sup, labels)
            loss_cls_inf = criterion(scores_inf, labels)
            total_loss = 0.5*loss_cls_sup - 0.5*loss_cls_inf
            total_loss.backward()
            self.masker_optim.step()

            self.global_step += 1

            # record
            self.logger.log(
                it=it,
                iters=len(self.train_loader),
                losses=loss_dict,
                samples_right=correct_dict,
                total_samples=num_samples_dict
            )

        # turn on eval mode
        self.encoder.eval()
        self.classifier.eval()
        self.masker.eval()
        self.classifier_ad.eval()

        # evaluation
        with torch.no_grad():
            for phase, loader in self.eval_loader.items():
                total = len(loader.dataset)
                class_correct = self.do_eval(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {'class': class_acc})
                self.results[phase][self.current_epoch] = class_acc

            # save from best model
            if self.results['test'][self.current_epoch] >= self.best_acc:
                self.best_acc = self.results['test'][self.current_epoch]
                self.best_epoch = self.current_epoch + 1
                self.logger.save_best_model(self.encoder, self.classifier, self.best_acc)

    def do_eval(self, loader):
        correct = 0
        for it, (batch, domain) in enumerate(loader):
            data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1
            features = self.encoder(data)
            scores = self.classifier(features)
            correct += calculate_correct(scores, labels)
        return correct


    def do_training(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)
        self.logger.save_config()

        self.epochs = self.config["epoch"]
        self.results = {"val": torch.zeros(self.epochs), "test": torch.zeros(self.epochs)}

        self.best_acc = 0
        self.best_epoch = 0

        for self.current_epoch in range(self.epochs):

            # step schedulers
            self.encoder_sched.step()
            self.classifier_sched.step()

            self.logger.new_epoch([group["lr"] for group in self.encoder_optim.param_groups])
            self._do_epoch()
            self.logger.finish_epoch()

        # save from best model
        val_res = self.results['val']
        test_res = self.results['test']
        self.logger.save_best_acc(val_res, test_res, self.best_acc, self.best_epoch - 1)

        return self.logger


def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, config, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
