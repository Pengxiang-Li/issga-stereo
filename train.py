from __future__ import print_function
import pdb
import argparse
import os
import random
import gc
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from models import hsm
import time
from utils import logger
from utils.loss_func import smooth_loss
import cv2
# import torch.cuda.amp
torch.backends.cudnn.benchmark=True


parser = argparse.ArgumentParser(description='HSM-Net')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--logname', default=time.asctime(),
                    help='log name')
parser.add_argument('--database', default='data',
                    help='data path')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train')
parser.add_argument('--start_epochs', type=int, default=1,
                    help='start number of epochs to train')
parser.add_argument('--batchsize', type=int, default=10,
                    help='samples per batch')
parser.add_argument('--loadmodel', default=None,help='weights path')
parser.add_argument('--savemodel', default='./log',
                    help='save path')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
torch.manual_seed(args.seed)

model = hsm(args.maxdisp,clean=False,level=1)
model = nn.DataParallel(model)
model.cuda()

# load model
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    dict_name = list(torch.load(args.loadmodel))
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if ('disp' not in k) }
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
SmoothLoss = smooth_loss()
def _init_fn(worker_id):
    np.random.seed()
    random.seed()
torch.manual_seed(args.seed)  # set again
torch.cuda.manual_seed(args.seed)


from dataloader import listfiles as ls
from dataloader import listsceneflow as lt
from dataloader import listflying3d as lf
from dataloader import listflying3dtest as lftest
from dataloader import KITTIloader2015 as lk15
from dataloader import KITTIloader2012 as lk12
from dataloader import MiddleburyLoader as DA
from dataloader import DrivingStereo as ds

batch_size = args.batchsize
scale_factor = args.maxdisp / 384. # controls training resolution
all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader('%s/carla-highres/trainingF'%args.database)
loader_carla = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,right_disparity=all_right_disp, rand_scale=[0.225,0.6*scale_factor], rand_bright=[0.8,1.2],order=2,data_aug = True)

all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader('%s/mb-ex-training/trainingF'%args.database)  # mb-ex
loader_mb = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,right_disparity=all_right_disp, rand_scale=[0.225,0.6*scale_factor], rand_bright=[0.8,1.2],order=0)

all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader('%s/mb-ex/trainingF'%args.database)  # mb-ex
loader_mb_ex = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,right_disparity=all_right_disp, rand_scale=[0.225,0.6*scale_factor], rand_bright=[0.8,1.2],order=0)

# flying 3d
all_left_img, all_right_img, all_left_disp, all_right_disp = lf.dataloader('%s/sceneflow/'%args.database)
loader_scene = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,right_disparity=all_right_disp, rand_scale=[0.9,2.4*scale_factor], order=2,data_aug = True)

all_left_img, all_right_img, all_left_disp, all_right_disp = lftest.dataloader('%s/sceneflow/'%args.database)
loader_scene_test = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,right_disparity=all_right_disp, rand_scale=[0.9,2.4*scale_factor], order=2,data_aug = False)

all_left_img, all_right_img, all_left_disp,_,_,_ = lk15.dataloader('%s/Kitti15/training/'%args.database,typ='train') # change to trainval when finetuning on KITTI
loader_kitti15 = DA.myImageFloder(all_left_img,all_right_img,all_left_disp, rand_scale=[0.9,2.4*scale_factor], order=0,data_aug = True)
 
all_left_img, all_right_img, all_left_disp= ds.dataloader('%s/driving_stereo/'%args.database,typ='train') # change to trainval when finetuning on KITTI
loader_ds = DA.myImageFloder(all_left_img,all_right_img,all_left_disp, rand_scale=[0.9,2.4*scale_factor], order=0,data_aug = True)

all_left_img, all_right_img, all_left_disp,_,_,_ = lk15.dataloader('%s/Kitti15/training/'%args.database,typ='trainval') # change to trainval when finetuning on KITTI
loader_kitti15_val= DA.myImageFloder(all_left_img,all_right_img,all_left_disp, rand_scale=[0.9,2.4*scale_factor], order=0,data_aug = False,training=False)

all_left_img, all_right_img, all_left_disp = lk12.dataloader('%s/Kitti12/training/'%args.database)
loader_kitti12 = DA.myImageFloder(all_left_img,all_right_img,all_left_disp, rand_scale=[0.9,2.4*scale_factor], order=0,data_aug = True)

all_left_img, all_right_img, all_left_disp, _ = ls.dataloader('%s/eth3d'%args.database)
loader_eth3d = DA.myImageFloder(all_left_img,all_right_img,all_left_disp, rand_scale=[0.9,2.4*scale_factor],order=0)
 

data_inuse = torch.utils.data.ConcatDataset([loader_scene])

TrainImgLoader = torch.utils.data.DataLoader(
         data_inuse, 
         batch_size= batch_size, shuffle= True, pin_memory=True, num_workers=batch_size, drop_last=True, worker_init_fn=_init_fn)

data_inuse_test = torch.utils.data.ConcatDataset([loader_scene_test])
TestImgLoader = torch.utils.data.DataLoader(
         data_inuse_test, 
         batch_size= 2, shuffle= False, drop_last=True, worker_init_fn=_init_fn)

print('%d batches per epoch'%(len(data_inuse)//batch_size))
# Mixed precision

def train(imgL,imgR,disp_L,vis_simiarlty):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_L = Variable(torch.FloatTensor(disp_L))

        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
 
        #---------
        mask = (disp_true > 0) & (disp_true < args.maxdisp)
        mask.detach_()

        optimizer.zero_grad()


        stacked,entropy = model(imgL,imgR,vis_simiarlty,ISSGA = True)

        disp8 = F.max_pool2d(disp_true.unsqueeze(1), 8, 8, 0, 1, False, False).squeeze(1) / 8
        disp16 = F.max_pool2d(disp_true.unsqueeze(1), 16, 16, 0, 1, False, False).squeeze(1) / 16
        disp32 = F.max_pool2d(disp_true.unsqueeze(1), 32, 32, 0, 1, False, False).squeeze(1) / 32
        disp64 = F.max_pool2d(disp_true.unsqueeze(1), 64, 64, 0, 1, False, False).squeeze(1) / 64

        mask8 = (disp8 < args.maxdisp/8) & (disp8 > 0)
        mask8.detach_()
        mask16 = (disp16 < args.maxdisp/16) & (disp16 > 0)
        mask16.detach_()
        mask32 = (disp32 < args.maxdisp/32) & (disp32 > 0)
        mask32.detach_()
        mask64 = (disp64 < args.maxdisp/64) & (disp64 > 0)
        mask64.detach_()

        lossweight = [16./85, 32./85 * 2, 32./85 * 4, 8./85, 32./85, 48./85,60./85 ]

        loss_list = [   F.smooth_l1_loss(stacked[0][mask], disp_true[mask], size_average=True) ,\
                        F.smooth_l1_loss(stacked[1][mask], disp_true[mask], size_average=True),\
                        F.smooth_l1_loss(stacked[2][mask], disp_true[mask], size_average=True),\
                        F.smooth_l1_loss(stacked[3][mask], disp_true[mask], size_average=True),\
                        F.smooth_l1_loss(stacked[4][mask], disp_true[mask], size_average=True),\
                        F.smooth_l1_loss(stacked[5][mask], disp_true[mask], size_average=True) ]

        lossdata_list = []
        loss =  loss_list[0] * lossweight[0] \
                + loss_list[1] * lossweight[1]\
                + loss_list[2] * lossweight[2]  \
                + loss_list[3] * lossweight[3]  \
                + loss_list[4] * lossweight[4] \
                + loss_list[5] * lossweight[5]#  

        lossdata_list.append(loss_list[0].data)
        for i in range(1, len(loss_list)):
            lossdata_list.append(loss_list[i].data)
        del loss_list
     
        loss.backward()
        optimizer.step()
 

        vis = {}
        if vis_simiarlty:
            vis['res 1'] = stacked[0].detach().cpu().numpy()
            vis['res 1/4'] = stacked[0].detach().cpu().numpy()
            vis['res 1/8'] = stacked[0].detach().cpu().numpy()
            vis['res 1/16'] = stacked[1].detach().cpu().numpy()
            vis['res 1/32'] = stacked[2].detach().cpu().numpy()
            vis['res 1/64'] = stacked[3].detach().cpu().numpy()
 
        lossvalue = loss.data
        del stacked
        del loss
       
        
 

        return lossvalue,vis,lossdata_list

def test(imgL,imgR,disp_L,vis_simiarlty):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_L = Variable(torch.FloatTensor(disp_L))

        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        #---------
        mask = (disp_true > 0) & (disp_true < args.maxdisp)
        mask.detach_()
        #----
 

        stacked,entropy = model(imgL,imgR,vis_simiarlty)
        pred = stacked
        
        epe = torch.mean(torch.abs(pred[mask]-disp_true[mask]))  
            # three px error
        loss_3px = ThreePxError(pred, disp_true ,mask)
        # print("data  epe: %.4f, loss_3:%.4f  " % (  epe, loss_3px,))
        del stacked
        del entropy
        epe_data, loss_3px_data = epe.data.cpu(), loss_3px.data.cpu()
        del epe
        del loss_3px
        return epe_data, loss_3px_data

def adjust_learning_rate(optimizer, epoch):
    
    if epoch == 1:
        lr = 1e-3
    elif epoch == 20:
        lr = 1e-4
    elif epoch==30: 
        lr = 1e-5 + 1e-5 + 1e-5
    elif epoch == 40:
        lr = 1e-6
    else:
        return None
    print("Setting learning rate: {} . ".format(lr))
    # print("optimizer.param_groups {}".format(optimizer.param_groups))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    log = logger.Logger(args.savemodel, name=args.logname)
    total_iters = 0
    #Mixed Precision
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.start_epochs, args.epochs+1):
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)
        vis_simiarlty = True
        ## training ##
        start_batch = time.time()
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):

            if total_iters == 0:
                print("------------Input image------------")
                print("imgL_crop shape {}".format(imgL_crop.shape))
                print("imgR_crop shape {}".format(imgR_crop.shape))
                print("disp_crop_L shape {}".format(disp_crop_L.shape))

            start_time = time.time() 
            loss,vis,lossdata_list = train(imgL_crop,imgR_crop, disp_crop_L,vis_simiarlty)
            end_time = time.time()

            print('Iter %d training loss = %.4f, time = %.2f' %(batch_idx, loss, end_time- start_time))
            print("MultiScale Loss: [1/8: {:.4f}] [1/16: {:.4f}] [1/32: {:.4f}] [1/64: {:.4f}]".format(lossdata_list[0], lossdata_list[1], lossdata_list[2], lossdata_list[3]))
            
            total_train_loss += loss

            vis_simiarlty = False
            if total_iters %10 == 0:
                log.scalar_summary('train/loss_batch',loss, total_iters)
                log.scalar_summary('train/loss_batch_64',lossdata_list[3], total_iters)
                log.scalar_summary('train/loss_batch_32',lossdata_list[2], total_iters)
                log.scalar_summary('train/loss_batch_16',lossdata_list[1], total_iters)
                log.scalar_summary('train/loss_batch_8',lossdata_list[0], total_iters)
                log.scalar_summary('train/loss_batch_1',lossdata_list[4], total_iters)
                log.scalar_summary('train/loss_batch_1_issga',lossdata_list[5], total_iters)
                # log.scalar_summary('train/smooth_loss',lossdata_list[-1], total_iters)
            if total_iters %50 == 49:
                vis_simiarlty = True
            if total_iters %50 == 0:
                log.image_summary('train/left', imgL_crop[0:1],total_iters)
                log.image_summary('train/right',imgR_crop[0:1],total_iters)
                log.image_summary('train--gt0',disp_crop_L[0:1],total_iters)
                log.histo_summary('train/gt_hist',np.asarray(disp_crop_L), total_iters)

                log.image_summary('train--res/ 1/8',vis['res 1/8'][0:1],total_iters)
                log.image_summary('train--res/ 1/16',vis['res 1/16'][0:1],total_iters)
                log.image_summary('train--res/ 1/32',vis['res 1/32'][0:1],total_iters)
                log.image_summary('train--res/ 1/64',vis['res 1/64'][0:1],total_iters)
               
            print("Epoch {} batch {} running time {:.2f}".format(epoch, batch_idx,time.time()-start_batch))
            print("---------------------------------")
            start_batch = time.time()
            total_iters += 1
 
            if (total_iters + 1)%1000==0:
                #SAVE
                savefilename = args.savemodel+'/'+args.logname+'/finetune_'+str(total_iters)+'_'+str(total_train_loss/batch_idx)+'.tar'
                torch.save({
                    'iters': total_iters,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/batch_idx,
                }, savefilename)
 
            
                 
        cv2.setNumThreads(0)        
        cv2.ocl.setUseOpenCL(False)

        log.scalar_summary('train/loss',total_train_loss/len(TrainImgLoader), epoch)
        gc.collect()
        torch.cuda.empty_cache()

        epe_list = []
        loss3px_list = []
        # testing
        print("test start after epoch {}".format(epoch))
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TestImgLoader):
    
            epe,loss3px = test(imgL_crop, imgR_crop, disp_crop_L, False)
            epe_list.append(epe)
            loss3px_list.append(loss3px)

            
        print('Test epe:',np.mean(epe_list))
        print('Test loss_3:',np.mean(loss3px_list))
        s = str(args.loadmodel)
        with open(os.path.join("./testlog/", "kitti_results.txt"), "a") as f :
            f.write("{} ---epoch {} \tmaxdisp:{} \tepe:{} \tloss_3:{}\% \t test model path:{} \r\n".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),epoch,
                                            str(args.maxdisp),
                                            str(np.nanmean(epe_list)),
                                            str(np.nanmean(loss3px_list)),
                                            args.loadmodel))

        print("test end after epoch {}".format(epoch))
        log.scalar_summary('test/epe',np.nanmean(epe_list), epoch)
        log.scalar_summary('test/loss3px',np.nanmean(loss3px_list), epoch)
        cv2.setNumThreads(0)        
        cv2.ocl.setUseOpenCL(False)
        torch.cuda.empty_cache()

def ThreePxError(pred,gt,mask):
    error_map=torch.abs(gt[mask]-pred[mask])
    correct=((error_map<3) | (error_map<(gt[mask]*0.05))) 
    correct=torch.sum(correct)
    total=torch.sum(mask)
    error_rate=1-(1.0*correct/total) 
    return error_rate*100

if __name__ == '__main__':
    main()
