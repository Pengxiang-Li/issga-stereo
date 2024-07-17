
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import pdb
import time
from .utils import * 

class similarity_measure(nn.Module):
    def __init__(self, inplanes):
        super(similarity_measure, self).__init__()
        self.conv0 = nn.Conv2d(32 + 2, 32, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu0 = nn.LeakyReLU(inplace=True)
        self.conv1 = conv2DBatchNorm(32, 16, 1, 1, 0)
        
        # self.conv1 = convbn(32, 16, 1, 1, 0, 1, 16)    
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = conv2DBatchNorm(16, 8, 1, 1, 0)
        # self.conv2 = convbn(16, 8, 1, 1, 0, 1, 8)
    
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):               
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='leaky_relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        
    def forward(self, x):
        # # print("x {}".format(x.shape))
        output = self.conv0(x)
        output = self.relu0(output)
        output = self.conv1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
       
        return output



def matrix_generation(scale=2):
    
    x=torch.arange(-scale//2,scale//2+1).float()
    x=torch.cat([x[:x.shape[0]//2],x[x.shape[0]//2+1:]]).unsqueeze(0)
    distance_matrix=x.expand(scale,scale).unsqueeze(0)
    
    distance_matrix=torch.cat([distance_matrix,distance_matrix.transpose(2,1)],0)
    distance_matrix=distance_matrix.unsqueeze(0)
    distance_matrix1=distance_matrix+0
    distance_matrix2=distance_matrix+0
    distance_matrix3=distance_matrix+0
    distance_matrix4=distance_matrix+0
    distance_matrix5=distance_matrix+0
    distance_matrix6=distance_matrix+0
    distance_matrix7=distance_matrix+0
    distance_matrix8=distance_matrix+0
    x=torch.arange(1,scale+1).float()
    x=x.expand(scale,scale).unsqueeze(0)
    #x=x.repeat(hr_feature.shape[0],hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
    distance_matrix1[:,0,:,:]=scale-x+1
    distance_matrix2[:,0,:,:]=x
    distance_matrix5[:,0,:,:]=distance_matrix2[:,0,:,:]
    distance_matrix6[:,0,:,:]=distance_matrix1[:,0,:,:]
    distance_matrix7[:,0,:,:]=distance_matrix2[:,0,:,:]
    distance_matrix8[:,0,:,:]=distance_matrix1[:,0,:,:]
    x=torch.arange(1,scale+1).float()
    x=x.expand(scale,scale).unsqueeze(0).transpose(2,1)
    
    distance_matrix3[:,1,:,:]=(scale-x+1)
    distance_matrix4[:,1,:,:]=x
    distance_matrix5[:,1,:,:]=distance_matrix3[:,1,:,:]
    distance_matrix6[:,1,:,:]=distance_matrix3[:,1,:,:]
    distance_matrix7[:,1,:,:]=distance_matrix4[:,1,:,:]
    distance_matrix8[:,1,:,:]=distance_matrix4[:,1,:,:]
    # print(distance_matrix3)
    
    return distance_matrix.cuda(),distance_matrix1.cuda(),distance_matrix2.cuda(),distance_matrix3.cuda(),distance_matrix4.cuda(), \
           distance_matrix5.cuda(),distance_matrix6.cuda(),distance_matrix7.cuda(),distance_matrix8.cuda()

class eight_related_context_mapping(nn.Module):
    def __init__(self, inplanes=32,scale=2):
        super(eight_related_context_mapping,self).__init__()
        self.similarity1 = similarity_measure(inplanes)
        self.sigmoid=nn.Sigmoid()
        self.distance_matrix,self.distance_matrix1,self.distance_matrix2,self.distance_matrix3,self.distance_matrix4, \
        self.distance_matrix5,self.distance_matrix6,self.distance_matrix7,self.distance_matrix8 = matrix_generation(scale)

    def forward(self, lr_feature, hr_feature, lr_feature_r, hr_feature_r, disp=192):
        with torch.no_grad():
            scale=hr_feature.shape[-1]//lr_feature.shape[-1]
            if scale%2!=0:
                exit()
            padding1=hr_feature[:,:1,:,:scale]*0-100
            padding2=hr_feature[:,:1,:scale,:]*0-100
            
            # position information
            distance_matrix=self.distance_matrix.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
            distance_matrix1=self.distance_matrix1.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
            distance_matrix2=self.distance_matrix2.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
            distance_matrix3=self.distance_matrix3.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
            distance_matrix4=self.distance_matrix4.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
            distance_matrix5=self.distance_matrix1.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
            distance_matrix6=self.distance_matrix2.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
            distance_matrix7=self.distance_matrix3.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
            distance_matrix8=self.distance_matrix4.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
            
            # alpha = 0.0001
            # distance_matrix,distance_matrix1,distance_matrix2,
            # distance_matrix3,distance_matrix4,distance_matrix5,distance_matrix6,
            # distance_matrix7,distance_matrix8 =  distance_matrix/distance_matrix.max() * alpha, distance_matrix1/distance_matrix1.max() * alpha,
            # distance_matrix2/distance_matrix2.max()* alpha, distance_matrix3/distance_matrix3.max()* alpha,
            # distance_matrix4/distance_matrix4.max()* alpha, distance_matrix5/distance_matrix5.max()* alpha, 
            # distance_matrix6/distance_matrix6.max()* alpha, distance_matrix7,distance_matrix7.max()* alpha,
            # distance_matrix8/distance_matrix8.max()* alpha
        # print("distance_matrix {}, hr_feature_r {}".format(distance_matrix.shape, hr_feature_r.shape))
        #ref img
            # lr_feature B C H W  ->  B C H*s W*s
        lr_feature=lr_feature.unsqueeze(-1).expand(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],lr_feature.shape[3],scale) \
            .contiguous().view(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],lr_feature.shape[3]*scale) \
            .unsqueeze(-2).expand(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],scale,lr_feature.shape[3]*scale) \
            .contiguous().view(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2]*scale,lr_feature.shape[3]*scale)

        #target image
            # lr_feature B C H W  ->  B C H*s W*s
        lr_feature_r=lr_feature_r.unsqueeze(-1).expand(lr_feature_r.shape[0],lr_feature_r.shape[1],lr_feature_r.shape[2],lr_feature_r.shape[3],scale) \
            .contiguous().view(lr_feature_r.shape[0],lr_feature_r.shape[1],lr_feature_r.shape[2],lr_feature_r.shape[3]*scale) \
            .unsqueeze(-2).expand(lr_feature_r.shape[0],lr_feature_r.shape[1],lr_feature_r.shape[2],scale,lr_feature_r.shape[3]*scale) \
            .contiguous().view(lr_feature_r.shape[0],lr_feature_r.shape[1],lr_feature_r.shape[2]*scale,lr_feature_r.shape[3]*scale)

            #cat in C channel B C H*s W*s -> B 2C H*s W*s
        #No 0 CENTER DIRECTION
        representation=torch.cat([lr_feature,hr_feature,distance_matrix],1)                       
        weight=self.similarity1(representation)
        fuse_weight = weight + 0
        fuse_weight1 = fuse_weight
        
        representation_target=torch.cat([lr_feature_r,hr_feature_r,distance_matrix],1)
        weight_target=self.similarity1(representation_target)

        #No 1 LEFT DIRECTION
        #ref img
            #cat in C channel B C H*s (W-1)*s -> B 2C H*s (W-1)*s (move "scale"*2 pixels to left)
        representation_l=torch.cat([lr_feature[:,:,:,:-scale],hr_feature[:,:,:,scale:],distance_matrix1[:,:,:,:-scale]],1)
        weight_l=self.similarity1(representation_l)
            #padding in W channel B C H*s (W-1)*s -> B 2C H*s W*s
        weight_l=torch.cat([padding1,weight_l],-1)
         #tar img
            #cat in C channel B C H*s (W-1)*s -> B 2C H*s (W-1)*s (move "scale"*2 pixels to left)
        representation_l_target=torch.cat([lr_feature_r[:,:,:,:-scale],hr_feature_r[:,:,:,scale:],distance_matrix1[:,:,:,:-scale]],1)
        weight_l_target=self.similarity1(representation_l_target)
        #padding in W channel B C H*s (W-1)*s -> B 2C H*s W*s
        weight_l_target=torch.cat([padding1,weight_l_target],-1)

        #No 2 RIGHT DIRECTION
        #ref image
            #cat in C channel B C H*s (W-1)*s -> B 2C H*s (W-1)*s (move "scale"*2 pixels to right)
        representation_r=torch.cat([lr_feature[:,:,:,scale:],hr_feature[:,:,:,:-scale],distance_matrix2[:,:,:,scale:]],1)
        weight_r=self.similarity1(representation_r)
        weight_r=torch.cat([weight_r,padding1],-1)
        
        #target image
        representation_r_target=torch.cat([lr_feature_r[:,:,:,scale:],hr_feature_r[:,:,:,:-scale],distance_matrix2[:,:,:,scale:]],1)
        weight_r_target=self.similarity1(representation_r_target)
        weight_r_target=torch.cat([weight_r_target,padding1],-1)

        #No 3 TOP DIRECTION
        #ref image
            #cat in C channel B C (H-1)*s W*s -> B 2C (H-1)*s W*s (move "scale"*2 pixels to top)
        representation_t=torch.cat([lr_feature[:,:,:-scale,:],hr_feature[:,:,scale:,:],distance_matrix3[:,:,:-scale,:]],1)
        weight_t=self.similarity1(representation_t)
        weight_t=torch.cat([padding2,weight_t],-2)
        #target image
        # representation_t_target=torch.cat([lr_feature_r[:,:,:-scale,:],hr_feature_r[:,:,scale:,:]],1)
        # weight_t_target=self.similarity1(representation_t_target)
        # weight_t_target=torch.cat([padding2,weight_t_target],-2)

        #No 4 BOTTOM DIRECTION
        #ref image
            #cat in C channel B C (H-1)*s W*s -> B 2C (H-1)*s W*s (move "scale"*2 pixels to top)
        representation_b=torch.cat([lr_feature[:,:,scale:,:],hr_feature[:,:,:-scale,:],distance_matrix4[:,:,scale:,:]],1)
        weight_b=self.similarity1(representation_b)
        weight_b=torch.cat([weight_b,padding2],-2)

        
        #No 5 LEFT-TOP DIRECTION
        #ref image
            #cat in C channel B C (H-1)*s (W-1)*s -> B 2C (H-1)*s (W-1)*s (move "scale"*2 pixels to top, "scale"*2 to left )
        representation_lt=torch.cat([lr_feature[:,:,:-scale,:-scale],hr_feature[:,:,scale:,scale:],distance_matrix5[:,:,:-scale,:-scale]],1)
        weight_lt=self.similarity1(representation_lt)
            # padding
        weight_lt=torch.cat([padding2,torch.cat([padding1[...,scale:,:],weight_lt],-1)],-2)
        
        #No 6 RIGHT-TOP DIRECTION
        representation_rt=torch.cat([lr_feature[:,:,:-scale,scale:],hr_feature[:,:,scale:,:-scale],distance_matrix6[:,:,:-scale,scale:]],1)
        weight_rt=self.similarity1(representation_rt)
        weight_rt=torch.cat([padding2,torch.cat([weight_rt,padding1[...,scale:,:]],-1)],-2)\
        
        #No 7 LEFT-BOTTOM DIRECTION
        representation_lb=torch.cat([lr_feature[:,:,scale:,:-scale],hr_feature[:,:,:-scale:,scale:],distance_matrix7[:,:,scale:,:-scale]],1)
        weight_lb=self.similarity1(representation_lb)
        weight_lb=torch.cat([torch.cat([padding1[...,scale:,:],weight_lb],-1),padding2],-2)
        
        #No 8 LEFT-BOTTOM DIRECTION
        representation_rb=torch.cat([lr_feature[:,:,scale:,scale:],hr_feature[:,:,:-scale,:-scale],distance_matrix8[:,:,scale:,scale:]],1)
        weight_rb=self.similarity1(representation_rb)
        weight_rb=torch.cat([torch.cat([weight_rb,padding1[...,:-scale,:]],-1),padding2],-2)

        # ref weight
            # cat 9 *[B 2C H*s W*s] -> B 18C H*s W*s
        weight_all = torch.cat([weight,weight_l,weight_r,weight_t,weight_b,weight_lt,weight_rt,weight_lb,weight_rb],dim=1)
        # if torch.max(torch.abs(weight_all)) > 85 : #FOR DEBUG
        #     scale_allweight = (torch.max(torch.abs(weight_all))/60).detach()
        #     weight_all = weight_all/scale_allweight
        #     fuse_weight = fuse_weight/scale_allweight
        #     #normlization
        weight_norm=F.softmax(weight_all, dim=1)
        # tar weight
            # cat 3 *[B 2C H*s W*s] -> B 6C H*s W*s
        weight_all_target=torch.cat([weight_target,weight_l_target,weight_r_target],dim=1)
        weight_norm_target=F.softmax(weight_all_target, dim=1)
        

        # fuse_weight =weight_norm[:,0:1,...].unsqueeze(1)
        # fuse weight
        fuse_weight=torch.exp(fuse_weight)
        fuse_weight=(fuse_weight)/(torch.sum(torch.sum(fuse_weight.view(fuse_weight.shape[0],fuse_weight.shape[1],fuse_weight.shape[2],fuse_weight.shape[3]//scale,scale),dim=-1) \
            .view(fuse_weight.shape[0],fuse_weight.shape[1],fuse_weight.shape[2]//scale,scale,fuse_weight.shape[3]//scale),dim=-2) \
            .unsqueeze(-1).expand(fuse_weight.shape[0],fuse_weight.shape[1],fuse_weight.shape[2]//scale,fuse_weight.shape[3]//scale,scale) \
            .contiguous().view(fuse_weight.shape[0],fuse_weight.shape[1],fuse_weight.shape[2]//scale,fuse_weight.shape[3]) \
            .unsqueeze(-2).expand(fuse_weight.shape[0],fuse_weight.shape[1],fuse_weight.shape[2]//scale,scale,fuse_weight.shape[3]) \
            .contiguous().view(fuse_weight.shape[0],fuse_weight.shape[1],fuse_weight.shape[2],fuse_weight.shape[3]))
        
        # construct weight_all_volume [B D H*s W*s] 
        weight_all_volume = torch.ones((weight_norm_target.shape[0], disp, weight_norm_target.shape[2], weight_norm_target.shape[3]),
                                        dtype=weight_norm_target.dtype, device=weight_norm_target.device)
        weight_all_volume_top = torch.zeros((weight_norm_target.shape[0], disp, weight_norm_target.shape[2], weight_norm_target.shape[3]),
                                        dtype=weight_norm_target.dtype, device=weight_norm_target.device)
        weight_all_volume_bottom = torch.zeros((weight_norm_target.shape[0], disp, weight_norm_target.shape[2], weight_norm_target.shape[3]),
                                        dtype=weight_norm_target.dtype, device=weight_norm_target.device)
        for i in range(disp):
            if i > 0:
                weight_all_volume[:,i,:,i:]=weight_norm_target[:,0,:,:-i]
                weight_all_volume_top[:,i,:,i:]=weight_norm_target[:,1,:,:-i]
                weight_all_volume_bottom[:,i,:,i:]=weight_norm_target[:,2,:,:-i]
            else:
                weight_all_volume[:,0,...]=weight_norm_target[:,0,...]
                weight_all_volume_top[:,0,...]=weight_norm_target[:,1,...]
                weight_all_volume_bottom[:,0,...]=weight_norm_target[:,2,...]  
        # return [B C D H*s W*s] 
        return  weight_norm[:,0:1,...].unsqueeze(1), \
                weight_norm[:,1:2,...].unsqueeze(1), \
                weight_norm[:,2:3,...].unsqueeze(1), \
                weight_norm[:,3:4,...].unsqueeze(1), \
                weight_norm[:,4:5,...].unsqueeze(1),\
                weight_norm[:,5:6,...].unsqueeze(1), \
                weight_norm[:,6:7,...].unsqueeze(1), \
                weight_norm[:,7:8,...].unsqueeze(1), \
                weight_norm[:,8:9,...].unsqueeze(1), \
                weight_all_volume.unsqueeze(1), \
                weight_all_volume_top.unsqueeze(1), \
                weight_all_volume_bottom.unsqueeze(1),\
                fuse_weight1.unsqueeze(1)

def InterScaleOP1(cost, scale, weight_all_volume, weight_all_volume_top, weight_all_volume_bottom, fuse_weight,
                weight, weight_l, weight_r, weight_t, weight_b, weight_lt, weight_rt, weight_lb, weight_rb,last_layer=False):
    
    # expand W/scale*H/scale*D/scale to W*H*D
    
    cost=cost.unsqueeze(-1).expand(cost.shape[0],cost.shape[1],cost.shape[2],cost.shape[3],cost.shape[4],scale) \
                                    .contiguous().view(cost.shape[0],cost.shape[1],cost.shape[2],cost.shape[3],cost.shape[4]*scale) \
                                .unsqueeze(-2).expand(cost.shape[0],cost.shape[1],cost.shape[2],cost.shape[3],scale,cost.shape[4]*scale) \
                                .contiguous().view(cost.shape[0],cost.shape[1],cost.shape[2],cost.shape[3]*scale,cost.shape[4]*scale) \
                                .unsqueeze(-3).expand(cost.shape[0],cost.shape[1],cost.shape[2],scale,cost.shape[3]*scale,cost.shape[4]*scale) \
                                .contiguous().view(cost.shape[0],cost.shape[1],cost.shape[2]*scale,cost.shape[3]*scale,cost.shape[4]*scale)
    # cost = cost.repeat(1,1,scale,scale,scale)
    
    # message passing along disparity dimension
    if cost.shape[2]/weight_all_volume.shape[2] ==2 :
        weight_all_volume = torch.cat([weight_all_volume,weight_all_volume],dim = 2)
        weight_all_volume_top = torch.cat([weight_all_volume_top,weight_all_volume_top],dim = 2)
        weight_all_volume_bottom = torch.cat([weight_all_volume_bottom,weight_all_volume_bottom],dim = 2)
    
    cost_fuse = cost.add(cost.mul(weight_all_volume))
    cost_fuse[:,:,scale:,:,:] = cost_fuse[:,:,scale:,:,:].add(cost[:,:,:-scale,:,:].mul(weight_all_volume_top[:,:,scale:,:,:]))
    cost_fuse[:,:,:-scale,:,:] = cost_fuse[:,:,:-scale,:,:].add(cost[:,:,scale:,:,:].mul(weight_all_volume_bottom[:,:,:-scale,:,:]))
    cost_fuse = cost_fuse*fuse_weight
    
    # summarize the scale*scale region, W*H*D to W/scale * H/scale*D, and then again to W*H*D
    cost_fuse=torch.sum(torch.sum(cost_fuse.view(cost_fuse.shape[0],cost_fuse.shape[1],cost_fuse.shape[2],cost_fuse.shape[3],cost_fuse.shape[4]//scale,scale),dim=-1) \
                            .view(cost_fuse.shape[0],cost_fuse.shape[1],cost_fuse.shape[2],cost_fuse.shape[3]//scale,scale,cost_fuse.shape[4]//scale),dim=-2) \
                    .unsqueeze(-1).expand(cost_fuse.shape[0],cost_fuse.shape[1],cost_fuse.shape[2],cost_fuse.shape[3]//scale,cost_fuse.shape[4]//scale,scale) \
                    .contiguous().view(cost_fuse.shape[0],cost_fuse.shape[1],cost_fuse.shape[2],cost_fuse.shape[3]//scale,cost_fuse.shape[4]) \
                    .unsqueeze(-2).expand(cost_fuse.shape[0],cost_fuse.shape[1],cost_fuse.shape[2],cost_fuse.shape[3]//scale,scale,cost_fuse.shape[4]) \
                    .contiguous().view(cost_fuse.shape[0],cost_fuse.shape[1],cost_fuse.shape[2],cost_fuse.shape[3],cost_fuse.shape[4])

    cost_spatial = cost_fuse*weight
    
    cost_spatial[...,scale:] += cost_fuse[...,:-scale]*weight_l[...,scale:]

    cost_spatial[...,:-scale] += cost_fuse[...,scale:]*weight_r[...,:-scale]
    cost_spatial[...,scale:,:] += cost_fuse[...,:-scale,:]*weight_t[...,scale:,:]
    cost_spatial[...,:-scale,:] += cost_fuse[...,scale:,:]*weight_b[...,:-scale,:]
    
    cost_spatial[...,scale:,scale:] += cost_fuse[...,:-scale,:-scale]*weight_lt[...,scale:,scale:]
    cost_spatial[...,scale:,:-scale] += cost_fuse[...,:-scale,scale:]*weight_rt[...,scale:,:-scale]
    cost_spatial[...,:-scale,scale:] += cost_fuse[...,scale:,:-scale]*weight_lb[...,:-scale,scale:]

    cost_spatial[...,:-scale,:-scale] += cost_fuse[...,scale:,scale:]*weight_rb[...,:-scale,:-scale]

    
    
    return cost_spatial  

def InterScaleOP(cost, scale, weight_all_volume, weight_all_volume_top, weight_all_volume_bottom, fuse_weight,
                weight, weight_l, weight_r, weight_t, weight_b, weight_lt, weight_rt, weight_lb, weight_rb,last_layer=False):
    
    cost=cost.unsqueeze(-1).expand(cost.shape[0],cost.shape[1],cost.shape[2],cost.shape[3],cost.shape[4],scale) \
                        .contiguous().view(cost.shape[0],cost.shape[1],cost.shape[2],cost.shape[3],cost.shape[4]*scale) \
                    .unsqueeze(-2).expand(cost.shape[0],cost.shape[1],cost.shape[2],cost.shape[3],scale,cost.shape[4]*scale) \
                    .contiguous().view(cost.shape[0],cost.shape[1],cost.shape[2],cost.shape[3]*scale,cost.shape[4]*scale) \
                    .unsqueeze(-3).expand(cost.shape[0],cost.shape[1],cost.shape[2],scale,cost.shape[3]*scale,cost.shape[4]*scale) \
                    .contiguous().view(cost.shape[0],cost.shape[1],cost.shape[2]*scale,cost.shape[3]*scale,cost.shape[4]*scale)
    if cost.shape[2]/weight_all_volume.shape[2] ==2 :
        weight_all_volume = torch.cat([weight_all_volume,weight_all_volume],dim = 2)
        weight_all_volume_top = torch.cat([weight_all_volume_top,weight_all_volume_top],dim = 2)
        weight_all_volume_bottom = torch.cat([weight_all_volume_bottom,weight_all_volume_bottom],dim = 2)
 
    cost_fuse = cost.add(cost.mul(weight_all_volume))
    cost_fuse[:,:,scale:,:,:] = cost_fuse[:,:,scale:,:,:].add(cost[:,:,:-scale,:,:].mul(weight_all_volume_top[:,:,scale:,:,:]))
    cost_fuse[:,:,:-scale,:,:] = cost_fuse[:,:,:-scale,:,:].add(cost[:,:,scale:,:,:].mul(weight_all_volume_bottom[:,:,:-scale,:,:]))
    cost_fuse = cost_fuse * fuse_weight
    cost_fuse = cost.add(cost.mul(weight_all_volume))
    cost_fuse[:,:,scale:,:,:] = cost_fuse[:,:,scale:,:,:].add(cost[:,:,:-scale,:,:].mul(weight_all_volume_top[:,:,scale:,:,:]))
    cost_fuse[:,:,:-scale,:,:] = cost_fuse[:,:,:-scale,:,:].add(cost[:,:,scale:,:,:].mul(weight_all_volume_bottom[:,:,:-scale,:,:]))

    cost_fuse = cost_fuse*fuse_weight

    # summarize the scale*scale region, W*H*D to W/scale * H/scale*D, and then again to W*H*D

    former_shape =  list(cost_fuse.shape)
    new_shape = list(cost_fuse.shape[:3]) + list(dim //scale for dim in  cost_fuse.shape[3:]) +[scale,scale]

    # Reshape and interpolate the tensor
    cost_fuse = cost_fuse.view(*new_shape)
    cost_fuse = torch.sum(torch.sum(cost_fuse,dim=-1),dim=-1)
    cost_fuse = F.interpolate(cost_fuse, size=former_shape[2:], mode='nearest')

    # Restore the original shape
    cost_fuse = cost_fuse.view(*former_shape)


    cost_spatial = cost_fuse*weight

    cost_spatial[...,scale:] = cost_spatial[...,scale:].add(cost_fuse[...,:-scale].mul(weight_l[...,scale:]))
    cost_spatial[...,:-scale] += cost_fuse[...,scale:].mul(weight_r[...,:-scale])
    cost_spatial[...,scale:,:] += cost_fuse[...,:-scale,:].mul(weight_t[...,scale:,:])
    cost_spatial[...,:-scale,:] += cost_fuse[...,scale:,:].mul(weight_b[...,:-scale,:])
    cost_spatial[...,scale:,scale:] += cost_fuse[...,:-scale,:-scale].mul(weight_lt[...,scale:,scale:])
    cost_spatial[...,scale:,:-scale] += cost_fuse[...,:-scale,scale:].mul(weight_rt[...,scale:,:-scale])
    cost_spatial[...,:-scale,scale:] += cost_fuse[...,scale:,:-scale].mul(weight_lb[...,:-scale,scale:])
    cost_spatial[...,:-scale,:-scale] += cost_fuse[...,scale:,scale:].mul(weight_rb[...,:-scale,:-scale])

 
    return cost_spatial  
    # summarize the scale*scale region, W*H*D to W/scale * H/scale*D, and then again to W*H*D

    return cost_spatial  

class refinement(nn.Module):
    def __init__(self, inplanes):
        super(refinement, self).__init__()
        self.conv1 = conv2DBatchNormRelu(1, 16, 1, 1, 0)
        self.conv2 = conv2DBatchNormRelu(16, 32, 1, 1, 0)
        self.conv3 = conv2DBatchNormRelu(32, 16, 1, 1 ,0)
        self.conv4 = conv2DBatchNormRelu(16, 1, 1, 1, 0)
        
    def forward(self, x):

        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
       
        return output
