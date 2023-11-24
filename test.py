# -*- codeing = utf-8 -*-
# @Time : 2023/10/26 22:58
# @Author : 李昌杏
# @File : extract_feature.py
# @Software : PyCharm
import argparse
import multiprocessing
import torch
import torch.utils.data
from joblib import Parallel, delayed
from torch import nn
from torch.utils.data import DataLoader
import os

from tqdm import tqdm

import network.student as vits
from datasets import load_data
import numpy as np
import torch.nn.functional as F

from network.teacher import ModalityFusionNetwork
from utils.utils import map_sake, prec_sake

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def compute_avg_prec(sketch_label, retrieval):
    num_correct = 0
    avg_prec = 0
    for photo_idx, photo_class in enumerate(retrieval, start=1):
        if photo_class == sketch_label:
            num_correct += 1
            avg_prec = avg_prec + (num_correct / photo_idx)

    if num_correct > 0:
        avg_prec = avg_prec / num_correct

    return avg_prec

def mAPcuda(photo_loader, sketch_loader, model_img,model_skt ,k=None):
    num_cores = min(multiprocessing.cpu_count(), 32)
    gallery_reprs = []
    gallery_labels = []
    model_skt.eval()
    model_img.eval()
    with torch.no_grad():
        for photo, label in photo_loader:
            photo, label = photo.cuda(), label.cuda()
            photo_reprs = model_img.embedding(photo)
            gallery_reprs.append(photo_reprs)
            gallery_labels.append(label)

        gallery_reprs = F.normalize(torch.cat(gallery_reprs))
        gallery_labels = torch.cat(gallery_labels)

        aps_all = []
        for sketch, label in sketch_loader:
            sketch, label = sketch.cuda(), label
            sketch_reprs = F.normalize(model_skt.embedding(sketch))
            # sketch_reprs = model.embedding(sketch,'skt')[2]
            ranks = torch.argsort(torch.matmul(sketch_reprs, gallery_reprs.T), dim=1, descending=True)
            # num_correct = torch.sum(gallery_labels[ranks[:, 0]] == label).item()
            retrievals = gallery_labels[ranks]
            if k is not None:
                retrievals = gallery_labels[ranks[:, :k]]

            aps = Parallel(n_jobs=num_cores)(
                delayed(compute_avg_prec)(label[sketch_idx].item(), retrieval.cpu().numpy()) for sketch_idx, retrieval
                in enumerate(retrievals))
            aps_all.extend(aps)

        return np.mean(aps_all)

def mAP1(photo_loader, sketch_loader, model_img,model_skt,k):
    gallery_reprs = []
    gallery_reprs_skt = []
    gallery_labels = []
    gallery_labels_skt = []
    model_skt.eval()
    model_img.eval()
    with torch.no_grad():
        for idx,(photo, label) in enumerate(tqdm(photo_loader)):
            photo, label = photo.cuda(), label
            photo_reprs = model_img.embedding(photo).cpu()
            gallery_reprs.append(photo_reprs)
            gallery_labels.append(label)

        gallery_reprs = F.normalize(torch.cat(gallery_reprs))
        gallery_labels = torch.cat(gallery_labels)

        for idx,(sketch, label) in enumerate(tqdm(sketch_loader)):
            sketch, label = sketch.cuda(), label
            sketch_reprs = F.normalize(model_skt.embedding(sketch)).cpu()
            gallery_reprs_skt.append(sketch_reprs)
            gallery_labels_skt.append(label)

        gallery_reprs_skt = F.normalize(torch.cat(gallery_reprs_skt))
        gallery_labels_skt = torch.cat(gallery_labels_skt)

    test_features_img = nn.functional.normalize(gallery_reprs, dim=1, p=2)
    test_features_skt = nn.functional.normalize(gallery_reprs_skt, dim=1, p=2)
    ############################################################################
    # Step 2: similarity
    sim = torch.mm(test_features_skt, test_features_img.T)
    k = {'map': test_features_skt.shape[0], 'precision': k}
    ############################################################################
    # Step 3: evaluate
    aps = map_sake(test_features_img.numpy(), gallery_labels.numpy(),
                   test_features_skt.numpy(), gallery_labels_skt.numpy(),sim, k=k['map'])
    prec = prec_sake(test_features_img.numpy(), gallery_labels.numpy(),
                   test_features_skt.numpy(), gallery_labels_skt.numpy(),sim,k=k['precision'])
    print('map{}: {:.4f} prec{}: {:.4f}'.format(k['map'], np.mean(aps), k['precision'], prec))

def evaluate_teacher_class(args):
    datasets, sk_valid_data, im_valid_data = load_data(args)
    num_class = datasets.get_num_class()
    model= ModalityFusionNetwork(args.img_size, 3, feature_dim=args.teacher_out_dim, heads=3,encoder_backbone=args.teacher_encoder,
                                    num_class=num_class,
                                    checkpoint_path=args.teacher_pre_weight).cuda()
    model.load_state_dict(torch.load(f'../weights/teacher_{args.dataset}_good.pth'))
    model.eval()
    data_loader=DataLoader(datasets, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    acc_img=0
    acc_skt=0

    for data in data_loader:
        sk, im, sketch_neg, image_neg,label,label_neg= data
        sk, im,label = sk.cuda(), im.cuda(),label.cuda()

        photo1_cls,sketch1_cls,photo1_f,sketch1_f = model(im, sk)

        photo_out=torch.argmax(F.softmax(photo1_f,dim=1),dim=1)
        sketch_out=torch.argmax(F.softmax(sketch1_f,dim=1),dim=1)

        acc_skt+=sum(sketch_out==label)
        acc_img+=sum(photo_out==label)
    print(f'img class acc{acc_img/len(datasets)}\nskt class acc{acc_skt/len(datasets)}')

def evaluate_student(args):
    datasets, sk_valid_data, im_valid_data = load_data(args)
    model_skt = vits.Student_SKT(datasets.get_num_class(),checkpoint_path='../weights/vit.npz').cuda()
    model_img = vits.Student_IMG(datasets.get_num_class(),checkpoint_path='../weights/vit.npz').cuda()
    model_skt.load_state_dict(torch.load(f'weights/student_{args.dataset}_skt_good.pth'))
    model_img.load_state_dict(torch.load(f'weights/student_{args.dataset}_img_good.pth'))

    skt_loader = DataLoader(sk_valid_data, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True)
    img_loader = DataLoader(im_valid_data, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True)
    # print(f'pre@{args.k}:',mAP(img_loader,skt_loader ,model_img,model_skt,args.k=200))
    # print(f'mAP}:',mAP(img_loader,skt_loader ,model_img,model_skt))
    print(mAP1(img_loader,skt_loader,model_img,model_skt,args.k))

def valid(skt_loader, img_loader,model_img,model_skt):
    model_img.eval()
    model_skt.eval()
    acc= mAPcuda(img_loader, skt_loader, model_img,model_skt, k=100)
    model_img.train()
    model_skt.train()
    return acc

def mAP_visual(photo_loader, sketch_loader, model_img,model_skt ,k=10):
    import cv2
    gallery_reprs = []
    gallery_labels = []
    gallery_photo = []
    with torch.no_grad():
        for photo, label,org in photo_loader:
            photo, label = photo.cuda(), label.cuda()
            photo_reprs = model_img.embedding(photo)
            gallery_reprs.append(photo_reprs)
            gallery_labels.append(label)
            gallery_photo.append(org)
        gallery_reprs = F.normalize(torch.cat(gallery_reprs))
        gallery_labels = torch.cat(gallery_labels)
        gallery_photo=torch.cat(gallery_photo)

        for sketch, label,org in sketch_loader:
            sketch, label = sketch.cuda(), label.cuda()
            sketch_reprs = F.normalize(model_skt.embedding(sketch))
            ranks = torch.argsort(torch.matmul(sketch_reprs, gallery_reprs.T), dim=1, descending=True).cpu()
            retrievals_photo = gallery_photo[ranks[:, :k]]
            for sketch_idx, retrieval_photo in enumerate(retrievals_photo):
                s=org[sketch_idx]
                print(s.shape)
                p=retrieval_photo
                pic=[]
                pic.append(s)
                for _ in p:pic.append(_)
                img=np.concatenate(pic,axis=1)
                print(img.shape)
                cv2.imwrite(f'{sketch_idx}.png',np.array(img))

            return 0
        return 0

def visual_student(args):
    datasets, sk_valid_data, im_valid_data = load_data(args,org=True)
    model_skt = vits.Student_SKT(datasets.get_num_class(),checkpoint_path='../weights/vit.npz').cuda()
    model_img = vits.Student_IMG(datasets.get_num_class(),checkpoint_path='../weights/vit.npz').cuda()
    model_skt.load_state_dict(torch.load('weights/student_skt_good.pth'))
    model_img.load_state_dict(torch.load('weights/student_img_good.pth'))
    model_skt.eval()
    model_img.eval()
    skt_loader = DataLoader(sk_valid_data, batch_size=6, shuffle=True, num_workers=2, pin_memory=True)
    img_loader = DataLoader(im_valid_data, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    print(mAP_visual(img_loader,skt_loader ,model_img,model_skt,k=10))

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")
        # dataset
        parser.add_argument('--data_path', type=str, default="../datasets")
        parser.add_argument('--dataset', type=str, default='tu_berlin',
                            choices=['Sketchy', 'tu_berlin', 'Quickdraw','Sketchy25'])
        parser.add_argument('--dataset_len', type=int, default=149428)
        parser.add_argument('--test_class', type=str, default='test_class_tuberlin30',
                            choices=['test_class_sketchy25', 'test_class_sketchy21', 'test_class_tuberlin30', 'Quickdraw'])
        parser.add_argument('--testall', default=True, action='store_true', help='train/test scale')
        parser.add_argument('--k', default=200)
        parser.add_argument("--seed", default=1234)
        #train
        parser.add_argument("--img_size", default=224)
        parser.add_argument("--epoch", default=50)
        parser.add_argument("--warmup_epochs", default=5)
        parser.add_argument("--batch_size", default=64)
        parser.add_argument("--stu_batch_size", default=128)
        parser.add_argument("--lr", default=3e-6)
        parser.add_argument("--min_lr", default=1e-6)
        parser.add_argument("--weight_decay", default= 0.04)
        parser.add_argument("--weight_decay_end", default=0.4)
        parser.add_argument("--MOCO_K", default= 16384)
        #net
        parser.add_argument("--teacher_out_dim", default=768)
        parser.add_argument("--teacher_encoder", default='vit_base_patch16_224')
        parser.add_argument("--VIT_pre_weight", default='../weights/vit.npz')
        parser.add_argument("--teacher_pre_weight", default='../weights/vit.npz')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

if __name__ == '__main__':
    args=Option().parse()
    # evaluate_teacher_class(args)
    evaluate_student(args)
    # visual_student(args)
