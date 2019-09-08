# This program takes as input the interpolated boundaries from the svm program and gives the average boundary and its plot
# Again specify the start and end frame along with the directories

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams['axes.linewidth'] = 2
plt.tick_params(width=2,length=8)
plt.tick_params(which='minor',width=3,length=4)

start_frame = 2000
end_frame = 2101
#ignore_frame_fol1 = [59,89] #for scd. 
avg_lip_x = [0]
avg_lip_y = [0]
for iter1 in range(start_frame, end_frame):
#    if iter1 in ignore_frame_fol1:
#        continue
    frame_no = str(iter1)
    file_identity = "boundary/bnd_"+ frame_no + ".txt"
    xy_identity = np.loadtxt(file_identity)
    x_identity = xy_identity[:, 0]
    y_identity = xy_identity[:, 1]
    minpos = 0
    #performing the average
    for i in range(0, len(y_identity)):
        if y_identity[i] > 10:
            minpos = i
            break
    for i in range(0, len(y_identity)):
        if 0 < y_identity[i] < y_identity[minpos] and x_identity[i] >10: 
            minpos = i

    x_identity = np.roll(x_identity, -minpos)
    y_identity = np.roll(y_identity, -minpos)
    avg_lip_x += x_identity
    avg_lip_y += y_identity
    plt.plot(x_identity, y_identity, '.-')

avg_lip_x /= (end_frame - start_frame)
avg_lip_y /= (end_frame - start_frame)

#plotting and saving the average boundary

output_directory = "boundary/avg_" + str(start_frame) + "-" + str(end_frame-1) + ".txt"
np.savetxt(output_directory, np.column_stack((avg_lip_x, avg_lip_y)))
figure = "boundary/avg_plt-" + str(start_frame) + "-" + str(end_frame-1) + ".png"
plt.axis('equal')
plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')
plt.xticks([])
plt.yticks([])
plt.plot(avg_lip_x, avg_lip_y, 'ko-')
plt.savefig(figure)
plt.show()
