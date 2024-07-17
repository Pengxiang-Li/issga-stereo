import torch 
import torch.nn as nn
import torch.nn.functional as F
class smooth_loss(nn.Module):
    def __init__(self):
        super(smooth_loss,self).__init__()
    def abs_mean(self,x1,x2,mask):
        return torch.mean(torch.abs(x1[mask] - x2[mask]))
    def forward(self,x,mask):
        # right
        x_r = x[:,:,:-1]
        x_r = F.pad(x_r,(1,0),"constant",0)
        x_r[:,:,0] = x[:,:,0]  
        x_r = self.abs_mean(x, x_r, mask)

        # # left
        x_l = x[:,:,1:]
        x_l = F.pad(x_l,(0,1),"constant",0)
        x_l[...,:-1] = x[...,:-1]   
        x_l = self.abs_mean(x, x_l, mask)
        
        # # bottom
        x_b = x[:,:-1,:]
        x_b = F.pad(x_b,(0,0,1,0),"constant",0)
        x_b[:,0,:] = x[:,0,:]
        x_b = self.abs_mean(x, x_b, mask)
        
        # # top
        x_t = F.pad(x[:,1:,:],(0,0,0,1),"constant",0)
        x_t[:,-1,:] = x[:,-1,:]
        x_t = self.abs_mean(x, x_t, mask)

        # # diagonal
        x_rb = F.pad(x[:,:-1,:-1],(1,0,1,0),"constant",0)
        x_rb[:,0,:] = x[:,0,:]
        x_rb[:,:,0] = x[:,:,0]
        x_rb = self.abs_mean(x, x_rb, mask)

        x_lb = F.pad(x[:,:-1,1:],(0,1,1,0))
        x_lb[:,-1,:] = x[:,-1,:]
        x_lb[:,:,0] = x[:,:,0]
        x_lb = self.abs_mean(x, x_lb, mask)

        x_rt = F.pad(x[:,:-1,1:],(1,0,0,1))
        x_rt[:,0,:] = x[:,0,:]
        x_rt[:,-1,:] = x[:,-1,:]
        x_rt = self.abs_mean(x, x_rt, mask)

        x_lb = F.pad(x[:,1:,1:],(0,1,0,1))
        x_lb[:,-1,:] = x[:,-1,:]
        x_lb[:,0,:] = x[:,0,:]
        x_lb = self.abs_mean(x, x_lb, mask)

        loss = x_b + x_l + x_lb + x_r + x_rb + x_rt + x_t
        # loss = torch.abs(torch.mean(x_r))
        return loss/8.0

def main():
    s = smooth_loss()
    x =  torch.randn(2,6,4, requires_grad=True)
    mask = torch.ones(2,6,4)
    mask = torch.ones(2,6,4) == mask
    print(x[0])
    print(mask[0])
    loss = s(x,mask)
    print(loss)
    loss.backward()
    return 

# main()
