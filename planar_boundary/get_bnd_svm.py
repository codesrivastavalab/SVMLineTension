# This program determines the boundary of the transmembrane protein system one folder at a time.
# Based on the cutoff, there might be a few frames which do not give a proper almost-circular boundary. Make sure to ignore these frames in the LT
# calculations. Using cutoff = 35, the whole of folder 4 gives bad boundaries, hence it is ignored in my LT calculations
# Follow the same instructions as the previous SVM program

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.interpolate import splprep, splev

#folder_no = 450
start_frame = 100
end_frame = 201
gma = 0.02
reg_prm = 1000

for frame_no in range(start_frame, end_frame):
    filename = "data_file/frame-"+str(frame_no)
    xy_idn = np.loadtxt(filename)
    x_idn = xy_idn[:, 0]
    y_idn = xy_idn[:, 1]
    identity = xy_idn[:,2]
    x_max = np.max(x_idn)
    y_max = np.max(y_idn)
    x_min = np.min(x_idn)
    y_min = np.min(y_idn)

    identity = xy_idn[:, 2]

    xx, yy = np.meshgrid(np.linspace(-5, x_max, 250), np.linspace(-5, y_max, 250))
    clf = svm.SVC(gamma=gma)#using linear kernel

    X_idn = np.column_stack((x_idn, y_idn))
    Y_idn = identity

    clf.fit(X_idn, Y_idn)
    
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles='solid', colors='k')
    plt.clf()

    paths = contours.collections[0].get_paths()
    bnd_len = np.empty(len(paths))
    for i in range(len(paths)):
        bnd_len[i] = len(paths[i])
    bnd_len_srtd = bnd_len.argsort()[::-1][:2]
    path_number = bnd_len_srtd[0]
    boundary_contour = contours.collections[0].get_paths()[path_number]
    boundary_xy = boundary_contour.vertices
    boundary_x = boundary_xy[:, 0]
    boundary_y = boundary_xy[:, 1]

    for iterator in range(len(x_idn)):
        if Y_idn[iterator] == 0:
            plt.scatter(x_idn[iterator], y_idn[iterator], color='red', s=30)
        if Y_idn[iterator] == 1:
            plt.scatter(x_idn[iterator], y_idn[iterator], color='blue', s=30)


    bnd_plt = "boundary/bnd_" + str(frame_no) + ".png"
    plt.plot(boundary_x, boundary_y, 'k.-')
    plt.axis("scaled")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(bnd_plt)
    # plt.show()
    plt.clf()
    
    centerx = np.mean(boundary_x)
    centery = np.mean(boundary_y)
    boundary_x = boundary_x - centerx
    boundary_y = boundary_y - centery
    [tck, u] = splprep([boundary_x, boundary_y], s=0)
    new_points = splev(np.linspace(0, 1, 2000), tck)
    intrpltd_bdn = "boundary/bnd_" + str(frame_no) + ".txt"
    np.savetxt(intrpltd_bdn, np.column_stack((new_points[0], new_points[1])))
