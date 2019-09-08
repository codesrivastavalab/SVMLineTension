import numpy as np
import matplotlib.pyplot as plt
start_frame = 100
end_frame = 201

ign_frames = [1]
avg_lip_x = [0]
avg_lip_y = [0]

number_of_frames = end_frame - start_frame
for frame_no in range(start_frame, end_frame):
    if frame_no in ign_frames:
        number_of_frames -= 1
        continue
    filename = "boundary/bnd_"+str(frame_no)+".txt"
    xy_identity = np.loadtxt(filename)
    x_identity = xy_identity[:, 0] 
    y_identity = xy_identity[:, 1] 
    avg_lip_x += x_identity
    avg_lip_y += y_identity
    plt.plot(x_identity, y_identity, '.-')

avg_lip_x /= number_of_frames
avg_lip_y /= number_of_frames

output_directory = "boundary/avg_" + str(start_frame) + "-" + str(end_frame-1) + ".txt"
np.savetxt(output_directory, np.column_stack((avg_lip_x, avg_lip_y)))
figure = "boundary/avg_plt_" + str(start_frame) + "-" + str(end_frame-1) + ".png"
plt.axis('equal')
plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')
plt.xticks([])
plt.yticks([])
plt.plot(avg_lip_x, avg_lip_y, 'ko-')
plt.savefig(figure)
plt.show()
