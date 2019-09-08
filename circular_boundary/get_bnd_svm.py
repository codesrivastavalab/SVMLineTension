# This program uses a support vector machine (svm) to determine the boundary of a phase separated lipid system. The sklearn package is used here
# This program is specifically written for the DUPC/DPPC/CHOL system from the paper "Computer simulations of the phase separation in model membranes", by
# Svetlana Baoukina et al., but can be modified for any other system with a closed boundary
# Input: files containing coordinates of the lipid heads along with 0/1 as an identifier for the lipid type
# Output: saves the coordinates of the boundary, a plot of the system along with the boundary and the interpolated boundaries, shifted to be centered
# around the origin
# The program assumes that all the input files are in the same directory and gives the output for all the frames from start_frame to end_frame
# Don't forget to change the directories for input and output files

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.interpolate import splprep, splev
start_frame = 2000
end_frame = 2101
# end_frame = start_frame + 1

x_shift = 0
y_shift = 0
# To specify the area and the number of points which the svm will be working with; 500 grid points are enough. I've used nm units everywhere
x_grid_min = -1
x_grid_max = 35
y_grid_min = -1
y_grid_max = 35
grid_points = 500
# svm parameters - rule of thumb: gamma<1, c>100; higher the gamma, higher the "complexity" of boundary,
# Intuitively, the gamma parameter defines how far the influence of a single training example reaches, with low values
# meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the inverse of the radius of
# influence of samples selected by the model as support vectors;
# The C parameter trades off misclassification of training examples against simplicity of the decision surface. A low C
# makes the decision surface smooth, while a high C aims at classifying all training examples correctly by giving the
# model freedom to select more samples as support vectors
# Above explanation are copied from the sklearn user guide
svm_gamma = 0.05
svm_c = 1000
# number of points in the interpolated boundary:
number_of_points = 2000

for frame_no in range(start_frame, end_frame):
    
    filename="data_file/xyz-"+str(frame_no)+".dat"
    xy_idn = np.loadtxt(filename)
    x_idn = xy_idn[:, 0]
    y_idn = xy_idn[:, 1]
    for i in range(len(y_idn)):
        if y_idn[i] >= 290: #230 for frames between 1000-1100 . for rest 290
            y_idn[i] = y_idn[i] - np.max(y_idn)
        if x_idn[i] >= 360: #360 for frames between 1000-1100 . for rest 330
            x_idn[i] = x_idn[i] - np.max(x_idn)
    
    x_max = np.max(x_idn)
    y_max = np.max(y_idn)
    x_min = np.min(x_idn)
    y_min = np.min(y_idn)
    
    chi_sqr = xy_idn[:, 2]



    xx, yy = np.meshgrid(np.linspace(-10, x_max, 400), np.linspace(-100, y_max, 400))
    # xx, yy = np.meshgrid(np.linspace(-52, 52, 500), np.linspace(-52, 52, 500))
    clf = svm.SVC(kernel='rbf', gamma=svm_gamma, C=svm_c)

    X_idn = np.column_stack((x_idn, y_idn))
    Y_idn = chi_sqr

    clf.fit(X_idn, Y_idn)
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles='solid', colors='k')
    plt.clf()
    # plt.clabel(contours, inline=1, fontsize=10)
    # plt.scatter(x_chi, y_chi, c=Y_chisqr)
    # plt.axis("scaled")
    # plt.show()

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
    #intrpltd_bdn = "/home/sahithya/Documents/archit/dupc-dppc-chol/Output/dupc_dppc_sah_hex/ dudp_chi_int_bnd_" + str(frame_no) + ".txt"
    np.savetxt(intrpltd_bdn, np.column_stack((new_points[0], new_points[1])))
    

# make sure the directories exist before running the code
#    output_directory = "boundary/bnd_" + frame_no + ".txt"
#    figure = "boundary/bnd_" + frame_no + ".png"
#    np.savetxt(output_directory, np.column_stack((boundary_x, boundary_y)))
#    plt.axis("equal")
#    plt.axis("scaled")
#    plt.plot(boundary_x, boundary_y, 'k.-')
#    plt.savefig(figure)
#    plt.show()

#    centerx = np.mean(boundary_x)
#    centery = np.mean(boundary_y)
#    boundary_x = boundary_x - centerx
#    boundary_y = boundary_y - centery
#    # [tck, u] = splprep([x_identity, y_identity], s=0)
#    [tck, u] = splprep([boundary_x, boundary_y], s=0)
#    new_points = splev(np.linspace(0, 1, number_of_points), tck)
#    output_directory_int = "boundary/intrpltd_bnd_" + frame_no + ".txt"
#    figure_int = "boundary/intrpltd_bnd_" + frame_no + ".png"
#    np.savetxt(output_directory_int, np.column_stack((new_points[0], new_points[1])))
#    plt.axis("equal")
#    plt.axis("scaled")
#    plt.plot(new_points[0], new_points[1], '.')
#    plt.savefig(figure_int)
    #plt.show()
