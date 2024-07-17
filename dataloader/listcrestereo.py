import os
def dataloader(path = '/data/crestereo',train=False,fg = False):
    dirs = os.listdir(path)
    print(dirs)
    files = []
    for dir in dirs:
        sub_path = os.path.join(path,dir)
        if os.path.isdir(sub_path):
            sub_files = os.listdir(sub_path)
            sub_files = [os.path.join(sub_path, x) for x in sub_files]
            files = files + sub_files


    left_files = [x for x in files if x[-8:-4] == 'left']
    right_files = [x for x in files if x[-9:-4] == 'right']
    disp_files = [x for x in files if x[-8:-4] == 'disp']

    left_files = sorted(left_files)
    right_files = sorted(right_files)
    disp_files = [x[:-4] + '.disp.png' for x in left_files]
    
    return left_files,right_files,disp_files,None
