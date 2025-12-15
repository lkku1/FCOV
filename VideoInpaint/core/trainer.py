import os
import cv2
import time
import math
import glob
from tqdm import tqdm
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
import torch.distributed as dist

from core.dataset import Dataset
from core.loss import AdversarialLoss
import loralib as lora


class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        if debug:
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # setup data set and data loader
        self.train_dataset = Dataset(config['data_loader'], split='train',  debug=debug)
        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'], 
                rank=config['global_rank'])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None), 
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)

        # set loss functions 
        self.adversarial_loss = AdversarialLoss()
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()

        # setup models including generator and discriminator
        net = importlib.import_module('model.'+config['model'])
        self.netG_DC = net.DepthCompletion().to(self.config['device'])
        self.netG_CR = net.ContentReconstruction().to(self.config['device'])
        self.netG_CE = net.ContentEnhancement().to(self.config['device'])

        data = torch.load(os.path.join(self.config["chickpoint"], 'stage1.pth'), map_location=self.config['device'])
        self.netG_DC.load_state_dict(data['netG'], strict=False)
        data = torch.load(os.path.join(self.config["chickpoint"], 'stage2.pth'), map_location=self.config['device'])
        self.netG_CR.load_state_dict(data['netG'], strict=False)
        data = torch.load(os.path.join(self.config["chickpoint"], 'stage3.pth'), map_location=self.config['device'])
        self.netG_CE.load_state_dict(data['netG'], strict=False)

        lora.mark_only_lora_as_trainable(self.netG_DC)
        lora.mark_only_lora_as_trainable(self.netG_CR)
        lora.mark_only_lora_as_trainable(self.netG_CE)

        for name, param in self.netG_DC.named_parameters():
            if ('pos_emb' in name.split(".")[-2]) :
                param.requires_grad = True
        for name, param in self.netG_CR.named_parameters():
            if ('pos_emb' in name.split(".")[-2]) :
                param.requires_grad = True 

        self.netD_S = net.SNTemporalPatchGANDiscriminator(3, 64)
        self.netD_S = self.netD_S.to(self.config['device'])
        self.netD_T = net.SNTemporalPatchGANDiscriminator(3, 64, conv_by= "2dtsm")
        self.netD_T = self.netD_T.to(self.config['device'])

        self.optimG_DC = torch.optim.Adam(
            filter(lambda p : p.requires_grad, self.netG_DC.parameters()), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimG_CR = torch.optim.Adam(
            filter(lambda p : p.requires_grad, self.netG_CR.parameters()), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimG_CE = torch.optim.Adam(
            filter(lambda p : p.requires_grad, self.netG_CE.parameters()), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))

        self.optimD_S = torch.optim.Adam(
            self.netD_S.parameters(), 
            lr=config['trainer']['lr'],
            amsgrad=True,
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimD_T = torch.optim.Adam(
            self.netD_T.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        # self.load()

        if config['distributed']:
            self.netG_DC = DDP(
                self.netG_DC, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netG_CR = DDP(
                self.netG_CR, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netG_CE = DDP(
                self.netG_CE, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)

            self.netD_S = DDP(
                self.netD_S, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            
            self.netD_T = DDP(
                self.netD_T, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    # get current learning rate
    def get_lr(self):
        return self.optimG_DC.param_groups[0]['lr']

     # learning rate scheduler, step
    def adjust_learning_rate(self):
        decay = 0.1**(min(self.iteration,self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG_DC.param_groups:
                param_group['lr'] = new_lr
        
            for param_group in self.optimG_CR.param_groups:
                param_group['lr'] = new_lr
        
            for param_group in self.optimG_CE.param_groups:
                param_group['lr'] = new_lr

            for param_group in self.optimD_S.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD_T.param_groups:
                param_group['lr'] = new_lr

    # add summary
    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name]/100, self.iteration)
            self.summary[name] = 0

    # load netG and netD
    def load(self):

        model_path = self.config['save_dir']

        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None
        if latest_epoch is not None:
            gen_dc_path = os.path.join(
                model_path, 'gen_dc_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_dc_path = os.path.join(
                model_path, 'opt_dc_{}.pth'.format(str(latest_epoch).zfill(5)))
            
            gen_cr_path = os.path.join(
                model_path, 'gen_cr_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_cr_path = os.path.join(
                model_path, 'opt_cr_{}.pth'.format(str(latest_epoch).zfill(5)))
            
            gen_ce_path = os.path.join(
                model_path, 'gen_ce_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_ce_path = os.path.join(
                model_path, 'opt_ce_{}.pth'.format(str(latest_epoch).zfill(5)))
            
            dis_s_path = os.path.join(
                model_path, 'dis_s_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_s_path = os.path.join(
                model_path, 'opt_s_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_t_path = os.path.join(
                model_path, 'dis_t_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_t_path = os.path.join(
                model_path, 'opt_t_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(model_path))
            data = torch.load(gen_dc_path, map_location=self.config['device'])
            self.netG_DC.load_state_dict(data['netG'], strict=False)
            data = torch.load(opt_dc_path, map_location=self.config['device'])
            self.optimG_DC.load_state_dict(data['optimG'])

            data = torch.load(gen_cr_path, map_location=self.config['device'])
            self.netG_CR.load_state_dict(data['netG'], strict=False)
            data = torch.load(opt_cr_path, map_location=self.config['device'])
            self.optimG_CR.load_state_dict(data['optimG'])

            data = torch.load(gen_ce_path, map_location=self.config['device'])
            self.netG_CE.load_state_dict(data['netG'], strict=False)
            data = torch.load(opt_ce_path, map_location=self.config['device'])
            self.optimG_CE.load_state_dict(data['optimG'])

            data = torch.load(dis_s_path, map_location=self.config['device'])
            self.netD_S.load_state_dict(data['netD'], strict=False)
            data = torch.load(opt_s_path, map_location=self.config['device'])
            self.optimD_S.load_state_dict(data['optimD'])

            data = torch.load(dis_t_path, map_location=self.config['device'])
            self.netD_T.load_state_dict(data['netD'], strict=False)
            data = torch.load(opt_t_path, map_location=self.config['device'])
            self.optimD_T.load_state_dict(data['optimD'])

            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            if self.config['global_rank'] == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

    # save parameters every eval_epoch
    def save(self, it):
        if self.config['global_rank'] == 0:
            gen_dc_path = os.path.join(
                self.config['save_dir'], 'gen_dc_{}.pth'.format(str(it).zfill(5)))
            opt_dc_path = os.path.join(
                self.config['save_dir'], 'opt_dc_{}.pth'.format(str(it).zfill(5)))
            
            gen_cr_path = os.path.join(
                self.config['save_dir'], 'gen_cr_{}.pth'.format(str(it).zfill(5)))
            opt_cr_path = os.path.join(
                self.config['save_dir'], 'opt_cr_{}.pth'.format(str(it).zfill(5)))
            
            gen_ce_path = os.path.join(
                self.config['save_dir'], 'gen_ce_{}.pth'.format(str(it).zfill(5)))
            opt_ce_path = os.path.join(
                self.config['save_dir'], 'opt_ce_{}.pth'.format(str(it).zfill(5)))
            
            dis_s_path = os.path.join(
                self.config['save_dir'], 'dis_s_{}.pth'.format(str(it).zfill(5)))
            opt_s_path = os.path.join(
                self.config['save_dir'], 'opt_s_{}.pth'.format(str(it).zfill(5)))
            dis_t_path = os.path.join(
                self.config['save_dir'], 'dis_t_{}.pth'.format(str(it).zfill(5)))
            opt_t_path = os.path.join(
                self.config['save_dir'], 'opt_t_{}.pth'.format(str(it).zfill(5)))
            

            # print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG_DC, torch.nn.DataParallel) or isinstance(self.netG_DC, DDP):
                netG_DC = self.netG_DC.module
                netG_CR = self.netG_CR.module
                netG_CE = self.netG_CE.module
                netD_S = self.netD_S.module
                netD_T = self.netD_T.module
            else:
                netG_DC = self.netG_DC
                netG_CR = self.netG_CR
                netG_CE = self.netG_CE

                netD_S = self.netD_S
                netD_T = self.netD_T
            torch.save({'netG': lora.lora_state_dict(netG_DC)}, gen_dc_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG_DC.state_dict()}, opt_dc_path)
            
            torch.save({'netG': lora.lora_state_dict(netG_CR)}, gen_cr_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG_CR.state_dict()}, opt_cr_path)
            
            torch.save({'netG': netG_CE.state_dict()}, gen_ce_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG_CE.state_dict(),
                        }, opt_ce_path)
            
            torch.save({'netG': netD_S.state_dict()}, dis_s_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimD': self.optimD_S.state_dict(),
                        }, opt_s_path)

            torch.save({'netG': netD_T.state_dict()}, dis_t_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimD': self.optimD_T.state_dict(),
                        }, opt_t_path)
            

            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))

    # train entry
    def train(self):
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
        
        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    # process input and calculate loss every training epoch
    def _train_epoch(self, pbar):
        device = self.config['device']

        for frames, depths, masks in self.train_loader:
            self.adjust_learning_rate()
            self.iteration += 1

            frames, depths, masks = frames.to(device), depths.to(device), masks.to(device)
            b, t, c, h, w = depths.size()
            masked_frames = (frames * (1 - masks).float())
            masked_depths = (depths * (1 - masks).float())
            pred_dep = self.netG_DC(masked_depths, masks)
            pred_dep = pred_dep.view(b, t, 1, h, w)

            pred_frame_init, enc_feat = self.netG_CR(masked_frames, pred_dep)
            pred_frame_init = pred_frame_init.view(b, t, 3, h, w)
            bt, c_, h_, w_ = enc_feat.shape
            enc_feat = enc_feat.view(b, t, c_, h_, w_)
      
            pred_frame_enhance = self.netG_CE(masked_frames + pred_frame_init * masks, enc_feat)
            pred_frame_enhance = pred_frame_enhance.view(b, t, 3, h, w)
            
            comp_frame_enhance = masked_frames + pred_frame_enhance * masks

            if self.config['global_rank'] == 0 and self.iteration % 500 == 0:
                frame = (frames[0, 0].permute(1, 2, 0).cpu().numpy() + 1) * 0.5
                pred = (comp_frame_enhance[0, 0].permute(1, 2, 0).detach().cpu().numpy() + 1) * 0.5
                mak = (masks[0,0,0].cpu().numpy())
                cv2.imwrite("1.png", frame * 255)
                cv2.imwrite("2.png", pred * 255)
                cv2.imwrite("3.png", mak*255)

            self.optimG_DC.zero_grad()
            self.optimG_CR.zero_grad()
            self.optimG_CE.zero_grad()

            gen_loss = 0
            
            # generator l1 loss
            dc_loss = self.l1_loss(pred_dep, depths)
        
            gen_loss += dc_loss * self.config["losses"]["depth_weight"]
            self.add_summary(
                self.gen_writer, 'loss/dc_loss', dc_loss.item())

            # generator l1 loss
            cr_loss = self.l1_loss(pred_frame_init, frames)
            gen_loss += cr_loss * self.config["losses"]["frame_init_weight"]
            self.add_summary(
                self.gen_writer, 'loss/cr_loss', cr_loss.item())
            
            # generator l1 loss
            ce_loss = self.l1_loss(pred_frame_enhance, frames)
            gen_loss += ce_loss * self.config["losses"]["frame_final_weight"] 
            self.add_summary(
                self.gen_writer, 'loss/ce_loss', ce_loss.item())
            
            dis_loss = 0

            # discriminator adversarial loss
            self.optimD_S.zero_grad()
            self.optimD_T.zero_grad()
            s_real_vid_feat = self.netD_S(frames)
            s_fake_vid_feat = self.netD_S(comp_frame_enhance.detach())
            s_dis_real_loss = self.adversarial_loss(s_real_vid_feat, True, True)
            s_dis_fake_loss = self.adversarial_loss(s_fake_vid_feat, False, True)
            dis_loss += (s_dis_real_loss + s_dis_fake_loss) / 2
            self.add_summary(
                self.dis_writer, 'loss/s_dis_vid_fake', s_dis_fake_loss.item())
            self.add_summary(
                self.dis_writer, 'loss/s_dis_vid_real', s_dis_real_loss.item())
            
            t_real_vid_feat = self.netD_T(frames)
            t_fake_vid_feat = self.netD_T(comp_frame_enhance.detach())
            t_dis_real_loss = self.adversarial_loss(t_real_vid_feat, True, True)
            t_dis_fake_loss = self.adversarial_loss(t_fake_vid_feat, False, True)
            dis_loss += (t_dis_real_loss + t_dis_fake_loss) / 2
            self.add_summary(
                self.dis_writer, 'loss/t_dis_vid_fake', t_dis_fake_loss.item())
            self.add_summary(
                self.dis_writer, 'loss/t_dis_vid_real', t_dis_real_loss.item())
            
            dis_loss.backward()
            self.optimD_S.step()
            self.optimD_T.step()

            # generator adversarial loss
            s_gen_vid_feat = self.netD_S(comp_frame_enhance)
            s_gan_loss = self.adversarial_loss(s_gen_vid_feat, True, False)
            s_gan_loss = s_gan_loss * self.config['losses']['loss_gan_spatial_weight']
            gen_loss += s_gan_loss
            self.add_summary(
                self.gen_writer, 'loss/s_gan_loss', s_gan_loss.item())
            
            t_gen_vid_feat = self.netD_T(comp_frame_enhance)
            t_gan_loss = self.adversarial_loss(t_gen_vid_feat, True, False)
            t_gan_loss = t_gan_loss * self.config['losses']['loss_gan_temporal_weight']
            gen_loss += t_gan_loss
            self.add_summary(
                self.gen_writer, 'loss/t_gan_loss', t_gan_loss.item())
            
            gen_loss.backward()

            self.optimG_DC.step()
            self.optimG_CR.step()
            self.optimG_CE.step()


            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    f"dcg: {dc_loss.item():.3f}; crg: {cr_loss.item():.3f};"
                    f"ceg: {ce_loss.item():.3f}; std: {gen_loss.item():.3f};"
                    f"dis: {dis_loss.item():.3f};"
                   
                    )
                )

            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
            if self.iteration > self.train_args['iterations']:
                break

