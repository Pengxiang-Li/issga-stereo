import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath,fgfilepath='',fg = False):

  classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
  print("classes {}".format(classes))
  image = [img for img in classes if img.find('frames_cleanpass') > -1]
  disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]
 
  test_left_img=[]

  test_right_img=[]
  test_left_disp = []
  test_right_disp = []


  flying_path = filepath + '/flying3d/frames_cleanpass'
  flying_disp = filepath + '/flying3d/disparity'
  flying_dir = flying_path+'/TEST/'

  subdir = ['A','B','C']

  for ss in subdir:
    flying = os.listdir(flying_dir+ss)

    for ff in flying:
      imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
      for im in imm_l:
       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
         test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

       test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
       test_right_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/right/'+im.split(".")[0]+'.pfm')

       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
         test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

  test_left_fg=[]
  if fg:
    
    flying_path_fg = fgfilepath + '/flying3d/frames_cleanpass'
    flying_dir_fg = flying_path_fg+'/TEST/'
    subdir = ['A','B','C']
    for ss in subdir:
      flying = os.listdir(flying_dir+ss)
      for ff in flying:
        imm_l = os.listdir(flying_dir_fg + ss+'/'+ff+'/left/')
        for im in imm_l:
          if is_image_file(flying_dir_fg+ss+'/'+ff+'/left/'+im):
            test_left_fg.append(flying_dir_fg+ss+'/'+ff+'/left/'+im)


  if fg:
    return test_left_img, test_right_img, test_left_disp, test_left_fg
  return test_left_img, test_right_img, test_left_disp, test_right_disp
