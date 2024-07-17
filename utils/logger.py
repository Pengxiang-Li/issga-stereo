from PIL import Image 
# import tensorflow as tf
from torch.autograd import Variable
import numpy as np
import scipy.misc
import os
import cv2
import torch
from torchvision import transforms


from torch.utils.tensorboard import SummaryWriter 
def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None])
        
    img_tensor = img_tensor.transpose(0,2).transpose(0,1)  # C x H x W  ---> H x W x C
    
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy()*255
    
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()
    
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
        
    return img
class Logger(object):

    def __init__(self, log_dir, name=None):
        """Create a summary writer logging to log_dir."""
        if name is None:
            name = 'temp'
        self.name = name
        if name is not None:
            try:
                os.makedirs(os.path.join(log_dir, name))
            except:
                pass
            # self.writer = tf.summary.create_file_writer(os.path.join(log_dir, name),
                                                # filename_suffix=name)
            self.writer = SummaryWriter(os.path.join(log_dir, name),filename_suffix=name)
        else:
            # self.writer = tf.summary.create_file_writer(log_dir, filename_suffix=name)
            self.writer = SummaryWriter(log_dir, filename_suffix=name)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, global_step=step, walltime=None)
        # summary = tf.summary.scalar(tag, value , step=step)
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary)
    
    def image_summary(self, tag, images, step):
        """Log a list of images."""
        # print("images {}".format(images.shape))
        # img_summaries = []
        for i, img in enumerate(images):
            min = img.min()
            img = img - min
            max = img.max()
            img = img / max
            
            if len(img.shape)== 3 & img.shape[0] !=1 :
                self.writer.add_image(str(tag) +"_"+ str(i),img , step)
            else:

                self.writer.add_image(str(tag) +"_"+ str(i), img, step,dataformats='HW')


    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # # Create a histogram using numpy
        # counts, bin_edges = np.histogram(values, bins=bins)

        # # Fill the fields of the histogram proto
        # hist = tf.HistogramProto()
        # hist.min = float(np.min(values))
        # hist.max = float(np.max(values))
        # hist.num = int(np.prod(values.shape))
        # hist.sum = float(np.sum(values))
        # hist.sum_squares = float(np.sum(values**2))

        # # Drop the start of the first bin
        # bin_edges = bin_edges[1:]

        # # Add bin edges and counts
        # for edge in bin_edges:
        #     hist.bucket_limit.append(edge)
        # for c in counts:
        #     hist.bucket.append(c)

        # # Create and write Summary
        # # summary = tf.summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        # summary = tf.summary.histogram(name = tag, data = hist)
        # self.writer.add_summary(summary, step)
        # self.writer.flush()

    def to_np(self, x):
        return x.data.cpu().numpy()

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def model_param_histo_summary(self, model, step):
        """log histogram summary of model's parameters
        and parameter gradients
        """
        for tag, value in model.named_parameters():
            if value.grad is None:
                continue
            tag = tag.replace('.', '/')
            tag = self.name+'/'+tag
            self.histo_summary(tag, self.to_np(value), step)
            self.histo_summary(tag+'/grad', self.to_np(value.grad), step)

