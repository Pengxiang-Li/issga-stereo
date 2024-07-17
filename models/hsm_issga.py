from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import time
from .submodule import *
from .issga import *
import pdb
from models.utils import unet
from matplotlib import pyplot as plt
 
 


class HSMNet(nn.Module):
    def __init__(self, maxdisp,clean,level=1):
        super(HSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.clean = clean
        self.feature_extraction = unet()
        self.level = level


        self.mapping_matrix   = eight_related_context_mapping(inplanes=32,scale=8)
        self.mapping_matrix8  = eight_related_context_mapping(inplanes=32,scale=2)
        self.mapping_matrix16 = eight_related_context_mapping(inplanes=32,scale=2)
        self.mapping_matrix32 = eight_related_context_mapping(inplanes=32,scale=2)
    
        # block 4
        self.decoder6 = decoderBlock(6,32,32,up=True, pool=True)
        if self.level > 2:
            self.decoder5 = decoderBlock(6,32,32,up=False, pool=True)
        else:
            self.decoder5 = decoderBlock(6,32,32,up=True, pool=True)
            if self.level > 1:
                self.decoder4 = decoderBlock(6,32,32, up=False)
            else:
                self.decoder4 = decoderBlock(6,32,32, up=True)
                self.decoder3 = decoderBlock(5,32,32, stride=(1,1,1),up=False, nstride=1)
  
        self.disp_reg = disparityregression(self.maxdisp,1)
        self.disp_reg8 = disparityregression(self.maxdisp,8)
        self.disp_reg8_1 = disparityregression(self.maxdisp,8)
        self.disp_reg16 = disparityregression(self.maxdisp,16)
        self.disp_reg32 = disparityregression(self.maxdisp,32)
        self.disp_reg64 = disparityregression(self.maxdisp,64)
   

    def feature_vol(self, refimg_fea, targetimg_fea,maxdisp, leftview=True):
        '''
        diff feature volume
        '''
        width = refimg_fea.shape[-1]
        maxdisp = int(maxdisp)
        cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], int(maxdisp),  refimg_fea.size()[2],  refimg_fea.size()[3]).fill_(0.))
        for i in range(min(maxdisp, width)):
            feata = refimg_fea[:,:,:,i:width]
            featb = targetimg_fea[:,:,:,:width-i]
            # concat
            if leftview:
                cost[:, :refimg_fea.size()[1], i, :,i:]   = torch.abs(feata-featb)
            else:
                cost[:, :refimg_fea.size()[1], i, :,:width-i]   = torch.abs(featb-feata)
        cost = cost.contiguous()
        return cost


    def forward(self, left, right,vis_simiarlty = False, ISSGA=True):
        scale = 2
        nsample = left.shape[0]
        conv4,conv3,conv2,conv1,conv0 = self.feature_extraction(torch.cat([left,right],0))
        conv40,conv30,conv20,conv10 ,conv00 = conv4[:nsample], conv3[:nsample], conv2[:nsample], conv1[:nsample],conv0[:nsample]
        conv41,conv31,conv21,conv11 ,conv01 = conv4[nsample:], conv3[nsample:], conv2[nsample:], conv1[nsample:],conv0[nsample:]

        feat6 = self.feature_vol(conv40, conv41, self.maxdisp//64)
        feat5 = self.feature_vol(conv30, conv31, self.maxdisp//32)
        feat4 = self.feature_vol(conv20, conv21, self.maxdisp//16)
        feat3 = self.feature_vol(conv10, conv11, self.maxdisp//8)

        weight,weight_l,weight_r,weight_t,weight_b,weight_lt,weight_rt,weight_lb,weight_rb, \
        weight_all_volume,weight_all_volume_top,weight_all_volume_bottom,fuse_weight \
            = self.mapping_matrix(conv10, conv00, conv10, conv00, disp=self.maxdisp)

        if ISSGA:
            # 1/16 scale -> 1/8 scale
            weight8,weight8_l,weight8_r,weight8_t,weight8_b,weight8_lt,weight8_rt,weight8_lb,weight8_rb, \
            weight8_all_volume,weight8_all_volume_top,weight8_all_volume_bottom,fuse_weight8 \
                = self.mapping_matrix8(conv20,conv10, conv21,conv11, disp=self.maxdisp//8)
            # weight8_list = [weight8,weight8_l,weight8_r,weight8_t,weight8_b,weight8_lt,weight8_rt,weight8_lb,weight8_rb, \
            #                     weight8_all_volume,weight8_all_volume_top,weight8_all_volume_bottom,fuse_weight8]
            
            # 1/32 scale -> 1/16 scale
            weight16,weight16_l,weight16_r,weight16_t,weight16_b,weight16_lt,weight16_rt,weight16_lb,weight16_rb, \
            weight16_all_volume,weight16_all_volume_top,weight16_all_volume_bottom,fuse_weight16 \
                = self.mapping_matrix16(conv30,conv20, conv31,conv21,disp=self.maxdisp//16)
            # weight16_list = [weight16,weight16_l,weight16_r,weight16_t,weight16_b,weight16_lt,weight16_rt,weight16_lb,weight16_rb, \
            #                 weight16_all_volume,weight16_all_volume_top,weight16_all_volume_bottom,fuse_weight16]
    
            # 1/64 scale -> 1/32 scale 
            weight32,weight32_l,weight32_r,weight32_t,weight32_b,weight32_lt,weight32_rt,weight32_lb,weight32_rb, \
            weight32_all_volume,weight32_all_volume_top,weight32_all_volume_bottom,fuse_weight32 = \
                self.mapping_matrix32(conv40[:,:16,...],conv30,conv41[:,:16,...],conv31, disp=self.maxdisp//32)
            # weight32_list = [weight32,weight32_l,weight32_r,weight32_t,weight32_b,weight32_lt,weight32_rt,weight32_lb,weight32_rb, \
            #                     weight32_all_volume,weight32_all_volume_top,weight32_all_volume_bottom,fuse_weight32]


        # t = time.time() -time_mark +t

        feat6_2x, cost6 = self.decoder6(feat6)
        feat5 = torch.cat((feat6_2x, feat5),dim=1)
        if ISSGA:
            time_mark = time.time()
            cost6_details = InterScaleOP(feat6, scale, weight32_all_volume, weight32_all_volume_top, weight32_all_volume_bottom, fuse_weight32,
                                        weight32, weight32_l, weight32_r, weight32_t, weight32_b, weight32_lt, weight32_rt, weight32_lb, weight32_rb)
            # print("cost6_spatial {}".format(cost6_spatial.shape))
            # feat5 = feat5 + cost6_spatial
            feat5 = feat5 + cost6_details
            # t = time.time() -time_mark +t
        # feat5_2x, cost5_1 = self.decoder5(cost6_spatial)
        feat5_2x, cost5 = self.decoder5(feat5)
        
        if self.level > 2:
            cost3 = F.upsample(cost5, [left.size()[2],left.size()[3]], mode='bilinear')
        else:
            feat4 = torch.cat((feat5_2x, feat4),dim=1)
            if ISSGA:
                time_mark = time.time()
                cost5_details = InterScaleOP(feat5, scale, weight16_all_volume, weight16_all_volume_top, weight16_all_volume_bottom, fuse_weight16,
                                        weight16, weight16_l, weight16_r, weight16_t, weight16_b, weight16_lt, weight16_rt, weight16_lb, weight16_rb)
                feat4 = feat4 + cost5_details
                # t = time.time() -time_mark +t

            feat4_2x, cost4 = self.decoder4(feat4)
            if self.level > 1:
                cost3 = F.upsample((cost4).unsqueeze(1), [self.disp_reg8.disp.shape[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
            else:
                feat3 = torch.cat((feat4_2x, feat3),dim=1)
                if ISSGA:
 
                    cost4_details =InterScaleOP(feat4, scale, weight8_all_volume, weight8_all_volume_top, weight8_all_volume_bottom, fuse_weight8,
                            weight8, weight8_l, weight8_r, weight8_t, weight8_b, weight8_lt, weight8_rt, weight8_lb, weight8_rb)   


    
                feat3_2x, cost3 = self.decoder3(feat3) # 32

                cost_spatial = InterScaleOP1(cost3.unsqueeze(1), 8, weight_all_volume, weight_all_volume_top, weight_all_volume_bottom, fuse_weight,
                                    weight, weight_l, weight_r, weight_t, weight_b, weight_lt, weight_rt, weight_lb, weight_rb).squeeze(1)
                

                cost3 = F.upsample(cost3, [left.size()[2],left.size()[3]], mode='bilinear')
                cost_bilinear = F.interpolate(cost3.unsqueeze(1), [self.maxdisp, left.size()[2], left.size()[3]],
                                        mode='trilinear', align_corners=True).squeeze(1)
                cost1 = cost_spatial + cost_bilinear
                t = time.time() -time_mark
                print("DDD time = {}".format(t*1000))
                    


        if self.level > 2:
            final_reg = self.disp_reg32
        else:
            final_reg = self.disp_reg8

        if self.training or self.clean==-1:
            pred3 = self.disp_reg8(F.softmax(cost3,1)); 
            pred1 = self.disp_reg(F.softmax(cost1,1))
            entropy = pred1
        else:
          
            pred1 = self.disp_reg(F.softmax(cost1,1))
            entropy = pred1


        if self.training:
            pred1_spatial = self.disp_reg(F.softmax(cost_spatial,1))
            pred6 = self.disp_reg64(F.softmax(cost6,1),mul=True)
            pred5 = self.disp_reg32(F.softmax(cost5,1),mul=True)
            pred4 = self.disp_reg16(F.softmax(cost4,1),mul=True)

            stacked = [pred3,pred4,pred5,pred6,pred1,pred1_spatial]   
            return stacked,entropy
        else:
            return pred1,entropy
