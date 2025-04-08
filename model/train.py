# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import time
import json
import joblib
import argparse
from util import *
import random
import numpy as np
import torch
from torch.optim import Adam, AdamW, SGD
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from model import *
from data import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reset_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args, ckp_dir, latest_checkpoint):
    """ Set seed """
    init_seed = args.seed + latest_checkpoint
    reset_seed(init_seed)
    """ Load data """
    train_dataset = Dataset(data_path=args.data_train, lpe_path=args.lpe_path, 
                            use_skeletal_feature=args.use_skeletal_feature, 
                            use_affective_feature=args.use_affective_feature)
    dev_dataset = Dataset(data_path=args.data_dev, lpe_path=args.lpe_path, 
                          use_skeletal_feature=args.use_skeletal_feature, 
                          use_affective_feature=args.use_affective_feature)
    test_dataset = Dataset(data_path=args.data_test, lpe_path=args.lpe_path, 
                           use_skeletal_feature=args.use_skeletal_feature, 
                           use_affective_feature=args.use_affective_feature)
    bone_size = train_dataset[0][1].shape[1]
    skeletal_dim = train_dataset[0][1].shape[2]
    affective_dim = train_dataset[0][2].shape[1]

    if args.use_skeletal_feature == 1 and args.use_affective_feature == 1:
        feat_dim = args.dim + affective_dim
    elif args.use_affective_feature == 1:
        feat_dim = affective_dim
    else:
        feat_dim = args.dim
    lpe_dim = train_dataset[0][3].shape[2]
    lp.lprint("------ Stats -----", True)
    lp.lprint("{:>21}:  {:,}".format("human bone size", bone_size), True)
    lp.lprint("{:>21}:  {:,}".format("skeletal dim", skeletal_dim), True)
    lp.lprint("{:>21}:  {:,}".format("affective dim", affective_dim), True)
    lp.lprint("{:>21}:  {:,}".format("feature dim", feat_dim), True)
    lp.lprint("{:>21}:  {:,}".format("LPE dim", lpe_dim), True)
    lp.lprint("{:>21}:  {:,}".format("training data size", len(train_dataset)), True)
    lp.lprint("{:>21}:  {:,}".format("development data size", len(dev_dataset)), True)
    lp.lprint("{:>21}:  {:,}".format("test data size", len(test_dataset)), True)
    """ Make dataloader """
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers, 
                                                   drop_last=False, 
                                                   collate_fn=collate_fn)
    dev_dataloader = torch.utils.data.DataLoader(dataset=dev_dataset, 
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers, 
                                                 drop_last=False, 
                                                 collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers, 
                                                  drop_last=False, 
                                                  collate_fn=collate_fn)
    """ Build the model """
    ske_encoder = SkeletalEncoder(dim=args.dim, bone_size=bone_size, skeletal_dim=skeletal_dim, lpe_dim=lpe_dim).to(device)
    ske_decoder = SkeletalDecoder(dim=args.dim, bone_size=bone_size, skeletal_dim=skeletal_dim, lpe_dim=lpe_dim).to(device)
    tempo_encoder = TemporalEncoder(input_dim=feat_dim, dim=args.dim).to(device)
    tempo_decoder = TemporalDecoder(output_dim=feat_dim, dim=args.dim).to(device)
    model = AE(ske_encoder, ske_decoder, tempo_encoder, tempo_decoder, args.use_skeletal_feature, args.use_affective_feature)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lp.lprint("{:>21}:  {:,}".format("model parameter size", params), True)

    """ Optimizer """
    if args.optim == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr)
    elif args.optim == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr*args.batch_size/256, weight_decay=0.0001, momentum=0.9)
    num_warmup_steps = len(train_dataloader) * args.warm_up_ratio
    num_training_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=num_warmup_steps,
                                                                   num_training_steps=num_training_steps,
                                                                   num_cycles=args.num_cycles)

    def train(epoch):
        model.train()
        sum_losses = AverageMeter()
        total = int(len(train_dataloader))
        """ Logging time """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        start = time.time()
        for i, (idxs, skes, afts, lpe, length) in enumerate(train_dataloader):
            """ Move to GPU """
            skes = skes.to(device)
            afts = afts.to(device)
            lpe = lpe.to(device)
            """ Zero Grad """
            optimizer.zero_grad()
            """ Calc Loss """
            loss = model(skes, afts, lpe, length)
            sum_losses.update(loss.item())
            """ Propagation """
            loss.backward()
            optimizer.step()
            """ Keep track of metrics """
            batch_time.update(time.time() - start)
            start = time.time()
            """ Print status """
            if i % args.log_interval == 0:
                lp.lprint('|      Training Epoch: {:4d}/{:4d}  '
                          '| Step:{:6d}/{:6d} '
                          '| lr:{:11.10f} '
                          '| {batch_time.avg:7.2f} sec/batch '
                          '| {data_time.avg:5.2f} sec/data_load '
                          '| Loss {loss.avg:5.6f} |'
                          .format(epoch+1, args.num_epochs, 
                                  i, total, 
                                  next(iter(optimizer.param_groups))['lr'], 
                                  batch_time=batch_time,
                                  data_time=data_time, 
                                  loss=sum_losses))
            """ update scheduler """
            scheduler.step()
        lp.lprint('|      Training Epoch: {:4d}/{:4d}  '
                  '| Step:{:6d}/{:6d} '
                  '| lr:{:11.10f} '
                  '| {batch_time.avg:7.2f} sec/batch '
                  '| {data_time.avg:5.2f} sec/data_load '
                  '| Loss {loss.avg:5.6f} |'
                  .format(epoch+1, args.num_epochs, 
                          i+1, total, 
                          next(iter(optimizer.param_groups))['lr'], 
                          batch_time=batch_time,
                          data_time=data_time, 
                          loss=sum_losses))
        return sum_losses.avg

    def validation(epoch):
        model.eval()
        sum_losses = AverageMeter()
        total = int(len(dev_dataloader))
        """ Logging time """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        start = time.time()
        for i, (idxs, skes, afts, lpe, length) in enumerate(dev_dataloader):
            """ Move to GPU """
            skes = skes.to(device)
            afts = afts.to(device)
            lpe = lpe.to(device)
            """ Calc Loss """
            loss = model(skes, afts, lpe, length)
            sum_losses.update(loss.item())
            """ Keep track of metrics """
            batch_time.update(time.time() - start)
            start = time.time()
            """ Print status """
            if i % args.log_interval == 0:
                lp.lprint('|    Validation Epoch: {:4d}/{:4d}  '
                          '| Step:{:6d}/{:6d} '
                          '| lr:{:11.10f} '
                          '| {batch_time.avg:7.2f} sec/batch '
                          '| {data_time.avg:5.2f} sec/data_load '
                          '| Loss {loss.avg:5.6f} |'
                          .format(epoch+1, args.num_epochs, 
                                  i, total, 
                                  next(iter(optimizer.param_groups))['lr'], 
                                  batch_time=batch_time,
                                  data_time=data_time, 
                                  loss=sum_losses))
        lp.lprint('|    Validation Epoch: {:4d}/{:4d}  '
                  '| Step:{:6d}/{:6d} '
                  '| lr:{:11.10f} '
                  '| {batch_time.avg:7.2f} sec/batch '
                  '| {data_time.avg:5.2f} sec/data_load '
                  '| Loss {loss.avg:5.6f} |'
                  .format(epoch+1, args.num_epochs, 
                          i+1, total, 
                          next(iter(optimizer.param_groups))['lr'], 
                          batch_time=batch_time,
                          data_time=data_time, 
                          loss=sum_losses))
        return sum_losses.avg

    def get_motion_vector(data_set, data_loader):
        output = []
        model.eval()
        total = int(len(data_loader))
        """ Logging time """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        start = time.time()
        for i, (idxs, skes, afts, lpe, length) in enumerate(data_loader):
            """ Move to GPU """
            skes = skes.to(device)
            afts = afts.to(device)
            lpe = lpe.to(device)
            """ Get vector """
            _, motion_vectors = model.get_vecs(skes, afts, lpe, length)
            """ Keep track of metrics """
            batch_time.update(time.time() - start)
            start = time.time()
            sidx = 0
            for j, idx in enumerate(idxs):
                _id = data_set.idx2id[idx]
                song_id = _id[0]
                start_frame = _id[1]
                eidx = sidx+length[j]
                motion_vector = motion_vectors[sidx:eidx].cpu().numpy()
                output.append({"song_id":song_id, "start_frame":start_frame, "motion_vector":motion_vector[:]})
                sidx += length[j]
            if i % args.log_interval == 0:
                lp.lprint('|   Get Motion Vector: {:6d}/{:6d} '
                          '| {batch_time.avg:7.2f} sec/batch '
                          '| {data_time.avg:5.2f} sec/data_load |'
                          .format(i, total, 
                                  batch_time=batch_time,
                                  data_time=data_time))
        lp.lprint('|   Get Motion Vector: {:6d}/{:6d} '
                  '| {batch_time.avg:7.2f} sec/batch '
                  '| {data_time.avg:5.2f} sec/data_load |'
                  .format(i+1, total, 
                          batch_time=batch_time,
                          data_time=data_time), True)
        return output

    def save_best_model(args):
        model.eval()
        with open(ckp_dir+"/skeletal_encoder_best.pt", 'wb') as f:
            torch.save(model.ske_encoder.state_dict(), f)            
        with open(ckp_dir+"/skeletal_decoder_best.pt", 'wb') as f:
            torch.save(model.ske_decoder.state_dict(), f)            
        with open(ckp_dir+"/tempo_encoder_best.pt", 'wb') as f:
            torch.save(model.tempo_encoder.state_dict(), f)            
        with open(ckp_dir+"/tempo_decoder_best.pt", 'wb') as f:
            torch.save(model.tempo_decoder.state_dict(), f)            
        with open(ckp_dir+"/model_best.pt", 'wb') as f:
            torch.save(model.state_dict(), f)            
        argparse_dict = vars(args)
        argparse_dict["best_ckp"] = best_ckp
        argparse_dict["best_loss"] = float(best_loss)
        with open(ckp_dir + "/model.param.json", 'w') as f:
            f.write(json.dumps(argparse_dict))
        with open(ckp_dir+"/scheduler.pt", 'wb') as f:
            torch.save(scheduler.state_dict(), f)            

    def load_best_model():
        if os.path.exists(ckp_dir+"/model_best.pt"):
            model.eval()
            model.load_state_dict(torch.load(ckp_dir+"/model_best.pt", weights_only=True, map_location=device))
            if os.path.exists(ckp_dir + "/scheduler.pt"):
                scheduler.load_state_dict(torch.load(ckp_dir + "/scheduler.pt", weights_only=True, map_location=device))

    """ Run """
    best_loss = 9999.9
    best_ckp = 0
    """ Load pre-trained model weight """
    if os.path.exists(ckp_dir + "/model_best.pt"):
        params = json.loads(open(ckp_dir + "/model.param.json").readline())
        best_ckp = params["best_ckp"]
        best_loss = params["best_loss"]
        lp.lprint("------ Load pre-trained encoder weight -----", True)
        lp.lprint("{:>21}:  {}".format("best ckp", best_ckp), True)
        lp.lprint("{:>21}:  {}".format("best loss", best_loss), True)
        load_best_model()
    lp.lprint("------ Training Encoder -----", True)
    no_update_times = 0
    for c, epoch in enumerate(range(args.num_epochs)):
        if c < latest_checkpoint:
            continue
        train_loss = train(epoch)
        lp.lprint("", True)
        with torch.no_grad():
            """ Validation """
            val_loss = validation(epoch)
            lp.lprint("", True)
            """ Check Loss """
            if val_loss < best_loss:
                best_loss = val_loss
                best_ckp = epoch+1
                lp.lprint("------ Update best model -----", True)
                lp.lprint("{:>21}:  {}".format("best ckp", best_ckp), True)
                lp.lprint("{:>21}:  {}".format("best loss", best_loss), True)
                save_best_model(args)
                no_update_times = 0
            else:
                no_update_times += 1
            if no_update_times >= args.early_stop:
                lp.lprint("Early Stoppimng %s"%(epoch+1), True)
                break
    lp.lprint("------ Get motion vector -----", True)
    load_best_model()
    with torch.no_grad():
        train_motion_vectors = get_motion_vector(train_dataset, train_dataloader)
        joblib.dump(train_motion_vectors, ckp_dir + "/motion_vector_train.joblib", compress=3)
        test_motion_vectors = get_motion_vector(test_dataset, test_dataloader)
        joblib.dump(test_motion_vectors, ckp_dir + "/motion_vector_test.joblib", compress=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Data """
    parser.add_argument("-data_train", "--data_train", dest="data_train", 
                        default="../data/instance/train.jsonl", 
                        type=str, help="training data")
    parser.add_argument("-data_dev", "--data_dev", dest="data_dev", 
                        default="../data/instance/dev.jsonl", 
                        type=str, help="development data")
    parser.add_argument("-data_test", "--data_test", dest="data_test", 
                        default="../data/instance/test.jsonl", 
                        type=str, help="development data")
    parser.add_argument("-lpe_path", "--lpe_path", dest="lpe_path", 
                        default="../data/lpe.joblib", 
                        type=str, help="Laplacian positional embedding file")

    """ Model parameter """
    parser.add_argument("-use_skeletal_feature", "--use_skeletal_feature", dest="use_skeletal_feature", 
                        default=1, type=int, help="Flag for using skeletal features")
    parser.add_argument("-use_affective_feature", "--use_affective_feature", dest="use_affective_feature", 
                        default=1, type=int, help="Flag for using affective features")
    parser.add_argument("-dim", "--dim", dest="dim", default=256, type=int, help="hidden dim")

    """ Training parameter """
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="random seed")
    parser.add_argument("-optim", "--optim", dest="optim", default="adamw", type=str, help="adam/adamw/sgd")
    parser.add_argument("-batch_size", "--batch_size", dest="batch_size", default=8, type=int, help="batch_size")
    parser.add_argument("-warm_up_ratio", "--warm_up_ratio", dest="warm_up_ratio", default=1.0, type=float, help="warm_up_ratio")
    parser.add_argument("-num_cycles", "--num_cycles", dest="num_cycles", default=2, type=int, help="num_cycles")
    parser.add_argument("-num_epochs", "--num_epochs", dest="num_epochs", default=200, type=int, help="Epochs")
    parser.add_argument("-lr", "--lr", dest="lr", default=0.00001, type=float, help="learning rate")
    parser.add_argument("-early_stop", "--early_stop", dest="early_stop", default=10, type=int, help="early stopping times")
    parser.add_argument("-num_workers", "--num_workers", dest="num_workers", default=8, type=int, help="number of CPU")
    parser.add_argument("-log_interval", "--log_interval", dest="log_interval", default=100, type=int, help="Report interval")

    """ Checkpoint parameter """
    parser.add_argument("-checkpoint", "--checkpoint", dest="checkpoint", default="./checkpoint", 
                        type=str, help="checkpoint directory")
    """ Logging parameter """
    parser.add_argument("-verbose", "--verbose", dest="verbose", default=1, type=int, help="verbose 0/1")
    args = parser.parse_args()

    """ Save parameter """
    ckp_dir = args.checkpoint.rstrip("/") + "/model_skeletal%s_aft%s_%s_lr%s"%(args.use_skeletal_feature, args.use_affective_feature, args.optim, args.lr)
    os.makedirs(ckp_dir, exist_ok=True)
    argparse_dict = vars(args)

    """ Check checkpoint directory """
    latest_checkpoint = 0
    if os.path.exists(ckp_dir + "/model.param.json"):
        params = json.loads(open(ckp_dir + "/model.param.json").readline())
        for k, v in params.items():
            if k == "best_ckp": 
                latest_checkpoint = int(v)
                continue
            if k == "best_loss": continue
            if k == "num_workers": continue
            if k == "verbose": continue
            if k == "log_interval": continue
            if k in argparse_dict:
                if argparse_dict[k] != v:
                    sys.stderr.write("ERROR: previous parameter %s (value: %s) is not specified value (%s) \n"%(k, argparse_dict[k], v))
                    exit(1)
            else:
                sys.stderr.write("ERROR: previous parameter dose not contain the key %s\n"%k)
                exit(1)
        if args.verbose == 1:
            lp = LogPrint(ckp_dir + "/model.log", True, 1)
        else:
            lp = LogPrint(ckp_dir + "/model.log", False, 1)
        lp.lprint("\n# continue training from checkpoint %03d"%latest_checkpoint, True)
    else:
        with open(ckp_dir + "/model.param.json", 'w') as f:
            f.write(json.dumps(argparse_dict))
        if args.verbose == 1:
            lp = LogPrint(ckp_dir + "/model.log", True)
        else:
            lp = LogPrint(ckp_dir + "/model.log", False)

    lp.lprint("------ Parameters -----", True)
    for k, v in argparse_dict.items():
        lp.lprint("{:>21}:  {}".format(k, v), True)
    main(args, ckp_dir, latest_checkpoint)


