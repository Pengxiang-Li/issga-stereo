import matplotlib.pyplot as plt
import random
import numpy as np

def vis_dis(z):
    plt.style.use('ggplot')
    
    x = np.arange(0, z.shape[0], 1)
    y = np.arange(0, z.shape[1], 1)
    x, y = np.meshgrid(y, x)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    print(x.shape,y.shape)
    plt.xlabel('Width')
    plt.ylabel('Dispatity')
    ax.set_zlabel('Probablity')
    ax.plot_surface(x, y, (z-z.min())/(z.max()-z.min()), rstride=1, cstride=1, cmap=plt.cm.jet)
    plt.savefig('./testcostpng/issga.png',dpi=600)
    # plt.show()

def vis_gt_surface(z,maxdisp = 192,line = 110):
    plt.style.use('ggplot')
    z_line=np.squeeze(z[line,:]).astype(int)
    print(z_line.shape)
    print(z_line.max())
    print(z_line)
    z_1 = np.zeros([maxdisp,z.shape[1] ],dtype=int,order='C')
    for i in range(z.shape[1]):
        # for j in range(maxdisp):
        z_1[z_line[i]][i]= 1
    x = np.arange(0, int(maxdisp), 1)
    y = np.arange(0, z.shape[1], 1)
    # x, y = np.meshgrid(y, x)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    print(x.shape,y.shape)
    plt.xlabel('Width')
    plt.ylabel('Dispatity')
    ax.set_zlabel('Probablity')
    print(x.shape,y.shape,z_1.shape)
    ax.plot_surface(x, y, (z_1-z_1.min())/(z_1.max()-z_1.min()), rstride=1, cstride=1,cmap=plt.cm.jet,alpha=0.6,  antialiased=False)

    plt.savefig('./testcostpng/gt3.png',dpi=600)
    # plt.show()

def vis_gt(z,maxdisp = 192,line = 110):
    plt.style.use('ggplot')
    z_line=np.squeeze(z[line,:]).astype(int)
    print(z_line.shape)
    print(z_line.max())
    print(z_line)
    z_1 = np.zeros([maxdisp,z.shape[1] ],dtype=int,order='C')
    for i in range(z.shape[1]):
        # for j in range(maxdisp):
        z_1[z_line[i]][i]= 1
    x = np.arange(0, z.shape[1], 1)
    y = z_line
    # x, y = np.meshgrid(y, x)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    print(x.shape,y.shape)
    plt.xlabel('Width')
    plt.ylabel('Dispatity')
    ax.set_zlabel('Probablity')
    print(x.shape,y.shape,z_1.shape)
    # plt.xlim((0, 960))
    plt.ylim((0, 200))
    # plt.zlim((0, 1.0))
    ax.set_zlim(0,1.0)
    ax.plot(x, y, zs=1.0, zdir='Probability')
    # ax.plot_surface(x, y, (z_1-z_1.min())/(z_1.max()-z_1.min()), rstride=1, cstride=1,cmap=plt.cm.jet,  antialiased=True)

    plt.savefig('./testcostpng/gt.png',dpi=600)
    # plt.show()

def vis_cost(cost,gt,name='issga_gt',maxdisp = 192,line = 110):
    plt.style.use('ggplot')
    z = cost
    z_line=np.squeeze(gt[line,:]).astype(int)
    x = np.arange(0, z.shape[0], 1)
    y = np.arange(0, z.shape[1], 1)
    x_gt = np.arange(0, gt.shape[1], 1)
    y_gt = z_line
    x, y = np.meshgrid(y,x)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # print(x.shape,y.shape)
    plt.xlabel('Width')
    plt.ylabel('Dispatity')
    ax.set_zlabel('Probablity')
    ax.set_zlim(0,1.0)
    ax.set_ylim(0,200)
    print(z.shape,name)
    ax.plot(x_gt, y_gt, zs=1.0)
    ax.plot_surface(x, y, (z-z.min())/(z.max()-z.min()), rstride=1, cstride=1, cmap=plt.cm.jet)
    plt.savefig('./testcostpng/%s.png' % name,dpi=600)


def vis_cost_surface(cost,gt,name='issga_gt',maxdisp = 192,line = 110):
    plt.style.use('ggplot')
    z = cost
    z_line=np.squeeze(gt[line-16,:]).astype(int)
    x = np.arange(0, z.shape[0], 1)
    y = np.arange(0, z.shape[1], 1)
    # z_1 = np.zeros([maxdisp,gt.shape[1] ],dtype=int,order='C')
    # print("z1 {}".format(z_1.shape))
    # for i in range(z.shape[1]):
        # z_1[z_line[i]//8-1][i]= 1
    x_gt = np.arange(0, gt.shape[1], 1)
    y_gt = z_line
    print(y_gt.shape)
    x, y = np.meshgrid(y,x)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # print(x.shape,y.shape)
    plt.xlabel('Width')
    plt.ylabel('Dispatity')
    ax.set_zlabel('Probablity')
    ax.set_zlim(0,1.0)
    ax.set_ylim(0,200)
    print(z.shape,name)
    ax.plot(x_gt, y_gt, zs=1.0, linewidth=0.5)
    # ax.scatter3D(x_gt, y_gt, zs=1.0)
    # ax.plot_surface(x, y, (z-z.min())/(z.max()-z.min()), rstride=1, cstride=1, cmap=plt.cm.jet,  antialiased=False)
    
    # ax.contour3D(x, y, (z_1-z_1.min())/(z_1.max()-z_1.min()),100, cmap='binary')
    ax.plot_surface(x, y, (z-z.min())/(z.max()-z.min()), rstride=1, cstride=1, cmap=plt.cm.jet,  antialiased=False)
    
    plt.savefig('./testcostpng/%s.png' % name,dpi=600)

def vis_cost_2d(cost,gt,name='issga_gt',maxdisp = 192,width=591, line = 110):
    y = cost[:,w]
    print(y.shape)
    x = np.arange(0, maxdisp, 1)
    gt_x = gt[line,w]

    # Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
    fig, ax = plt.subplots()
    ax.plot(x, (y-y.min())/(y.max()-y.min()))  # Plot some data on the axes.
    ax.scatter(gt_x,1.0)
    ax.set_xlabel('Disparity')  # Add an x-label to the axes.
    ax.set_ylabel('Probablity')  # Add a y-label to the axes.
    # ax.set_title("Simple Plot")  # Add a title to the axes.
    # ax.legend();  # Add a legend.
    plt.savefig('./testcostpng2d/%s.png' % name,dpi=600)

def vis_cost_2d_multi(bi,deconv,issga,gt,name='multi',maxdisp = 192,width=591, line = 110):
    x = np.arange(0, maxdisp, 1)
    w = width
    gt_x = gt[line,w]
    y_bi = bi[:,w]
    y_de = deconv[:,w]
    y_gu = issga[:,w]
    fig, ax = plt.subplots(1,3,figsize=(18,6))
    
    ax[0].plot(x, (y_bi-y_bi.min())/(y_bi.max()-y_bi.min()))
    ax[0].scatter(gt_x,1.0)
    ax[0].set_xlabel('Disparity')   
    ax[0].set_ylabel('Probablity')  
    ax[0].set_title("Bilinear") 

    ax[1].plot(x, (y_de-y_de.min())/(y_de.max()-y_de.min()))
    ax[1].scatter(gt_x,1.0)
    ax[1].set_xlabel('Disparity')   
    ax[1].set_ylabel('Probablity')  
    ax[1].set_title("Deconvolution") 

    ax[2].plot(x, (y_gu-y_gu.min())/(y_gu.max()-y_gu.min()))
    ax[2].scatter(gt_x,1.0)
    ax[2].set_xlabel('Disparity')   
    ax[2].set_ylabel('Probablity')  
    ax[2].set_title("Ours") 
    name = name + '_' + str(line) + '_' + str(w)
    plt.savefig('./testcostpng2d/%s.png' % name ,dpi=200)


    
