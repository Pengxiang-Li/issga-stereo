from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import time
from models import hsm
from utils import logger
from utils.readpfm import readPFM
import skimage.io
from PIL import Image
import cv2
from utils.preprocess import get_transform
import torchvision.transforms as transforms
 

parser = argparse.ArgumentParser(description='HSM-Net')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--logname', default=time.asctime(),
                    help='log name')
parser.add_argument('--database', default='/datsa',
                    help='data path')
parser.add_argument('--loadmodel',type=str, default=None,
                    help='weights path')
parser.add_argument('--testres', type=float, default=1,
                    help='test time resolution ratio 0-x')
parser.add_argument('--fg', action="store_true",
                    help='weights path')
args = parser.parse_args()
 

model = hsm(args.maxdisp,clean=False,level=1)
model = nn.DataParallel(model)
model.cuda()
torch.cuda.synchronize(device=0)
# load model
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    dict_name = list(torch.load(args.loadmodel))
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if ('disp' not in k) }
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
    print("model loaded from {}".format(args.loadmodel))
    for i,n in  enumerate(model.named_parameters()):
        print(i,n[0])
            # for i,p in  enumerate(model.parameters()):
    #     if i >=90 and i <=132 :
    #         continue
    #     elif i >=75 and i <= 77 :
    #         continue
    #     else:
    #         p.requires_grad  = False

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

from dataloader import listflying3dtest_fg as lftest
from dataloader import listflying3d as lftrain
from dataloader import KITTIloader2015 as lk15
# test_left_img, test_right_img, test_left_disp, test_left_fg = lftest.dataloader('%s/sceneflow/'%args.database,'%s/fine-grained-mask/sceneflow/'%args.database,fg=False)
test_left_img, test_right_img, test_left_disp,_,_,_ = lk15.dataloader('%s/Kitti15/training/'%args.database,typ='trainval') # change to trainval when finetuning on KITTI
def test(imgL,imgR,disp_L):
    model.train()

    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))   
    disp_L = Variable(torch.FloatTensor(disp_L))
    
    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    disp_true = disp_true.squeeze(0)

    #---------

    mask = (disp_true > 0) & (disp_true < args.maxdisp)  
    mask.detach_()
    #----

    
    stacked,entropy = model(imgL,imgR)
    pred = stacked[4]

    epe = torch.mean(torch.abs(pred[mask]-disp_true[mask]))  
    loss_3px = ThreePxError(pred, disp_true ,mask)

 
    epe_data, loss_3px_data = epe.data.cpu(), loss_3px.data.cpu()
    del epe
    del loss_3px

    return epe_data, loss_3px_data

 
def main():

    epe_list = []
    loss3px_list = []
    processed = get_transform()
    print('total test sample numbers: {}'.format(len(test_left_img)))
    print("Fine grained areas {}".format(args.fg))
    for inx in range(len(test_left_img)):
        # if inx%100 != 0:
        #     continue

        imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))[:,:,:3]
        imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))[:,:,:3]

        # resize
        imgL_o = cv2.resize(imgL_o,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_CUBIC)
        imgR_o = cv2.resize(imgR_o,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_CUBIC)
  
        
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
 

        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        top_pad = max_h-imgL.shape[2]
        left_pad = max_w-imgL.shape[3]
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        # load gt
        # gt = (skimage.io.imread().astype('float32')) 
        gt = Image.open(test_left_disp[inx])
        gt = np.ascontiguousarray(gt,dtype=np.float32)/256
        gt = np.reshape(gt,[1,1,gt.shape[0],gt.shape[1]])
        gt = np.lib.pad(gt,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    

        # test
        start = time.time()
        
        epe,loss3px= test(imgL, imgR, gt)
   

        print("Iter {}: epe: {:.4F} loss3px: {:.4F}. time: {:.2f}".format(inx,epe,loss3px,time.time() - start))
        if epe != float('nan'):
            epe_list.append(epe)
            loss3px_list.append(loss3px)
        
    print('Test epe:',np.nanmean(epe_list))
    print('Test loss_3:',np.nanmean(loss3px_list))
    model_path = str(args.loadmodel) 


    with open(os.path.join("./testlog/", "test_results.txt"), "a") as f :
        if args.fg:
            f.write("{} --- Fine grained areas: \tmaxdisp:{}  \tepe:{} \tloss_3:{}\% \t test model path: {} \r\n".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                                                str(args.maxdisp),
                                                str(np.nanmean(epe_list)),
                                                str(np.nanmean(loss3px_list)),
                                                args.loadmodel) )
        else:
            f.write("{} --- \tmaxdisp:{} \tepe:{} \tloss_3:{}\% \t test model path:{} \r\n".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                                                str(args.maxdisp),
                                                str(np.nanmean(epe_list)),
                                                str(np.nanmean(loss3px_list)),
                                                args.loadmodel))

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
