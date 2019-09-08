# This program calculates the fourier transform using the average file along with the individual frames

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams['axes.linewidth'] = 2
#plt.tick_params(width=3,length=8)
#plt.tick_params(which='minor',width=2,length=4)


start_frame = 100
end_frame = 201
number_of_frames = end_frame - start_frame
ignore_frame_fol1 = [1]
height_ft_avg = [0]
average_file = "boundary/avg_" + str(start_frame) + "-" + str(end_frame-1) + ".txt"
average_xy = np.loadtxt(average_file)
average_x = average_xy[:, 0]
average_y = average_xy[:, 1]
length = len(average_x)
perimeter = 0
for i in range(length - 1):
    perimeter += np.sqrt(np.square(average_x[i+1] - average_x[i]) + np.square(average_y[i+1] - average_y[i]))

for iterator in range(start_frame, end_frame):
    if iterator in ignore_frame_fol1:
        number_of_frames -= 1
        continue
    input_directory = "boundary/bnd_"+str(iterator)+".txt"
    datum = np.loadtxt(input_directory)
    frm_xcod = datum[:, 0]
    frm_ycod = datum[:, 1]

    inst_perimeter = 0    
    height_flcn = []
    for i1 in range(length):
        dif_of_angl = abs(average_y - frm_ycod[i1])
        min_angl_diff = np.min(dif_of_angl)
        posn_of_min = np.nonzero(dif_of_angl == min_angl_diff)[0][0]
        min_dist_sgnd = np.sqrt((((frm_xcod[i1]) - (average_x[posn_of_min])) ** 2) + (((frm_ycod[i1]) - (average_y[posn_of_min])) ** 2))
        height_flcn.append(min_dist_sgnd)
        inst_perimeter += np.sqrt(np.square(frm_xcod[i1-1] - frm_xcod[i1]) + np.square(frm_ycod[i1-1] - frm_ycod[i1]))
    outf1 = "spectrum/perimeter_" + str(start_frame) + "-" + str(end_frame) + ".txt"
    f= open(outf1,"a")
    f.write("%d %d \n" %(iterator, inst_perimeter))
        
    height_flcn_array = np.asarray(height_flcn)
    height_ft_avg += 2*(abs((np.fft.rfft(height_flcn_array)) / len(height_flcn_array))) ** 2

height_ft_avg /= number_of_frames
the_x = []
for itrr in range(len(height_ft_avg)):
    the_x.append((2*np.pi*itrr)/perimeter)

plt.loglog(the_x[2:],height_ft_avg[2:],'o')
plt.xlabel('k')
plt.ylabel('$ \langle|\delta R|^2 \\rangle$')
plt.show()
outf = "spectrum/spectrum_" + str(start_frame) + "-" + str(end_frame) + ".txt"
np.savetxt(outf, np.column_stack((the_x[1:], height_ft_avg[1:]*perimeter)))
