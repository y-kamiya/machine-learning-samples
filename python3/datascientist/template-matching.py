import sys
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

template_path = sys.argv[1]
target_path = sys.argv[2]

template = io.imread(template_path, as_grey = True)
target = io.imread(target_path, as_grey = True)
th, tw = template.shape

# save SSD on each point
score_map = np.zeros(shape = (target.shape[0] - th,
                              target.shape[1] - tw))

# calculate SSD by looking through the target
for y in range(score_map.shape[0]):
    for x in range(score_map.shape[1]):
        diff = target[y:y+th, x:x+tw] - template
        score_map[y, x] = np.square(diff).sum()

# extract point that have minimum SSD
x, y = np.unravel_index(np.argmin(score_map), score_map.shape)

# output
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8,3))
ax1.imshow(template, cmap=cm.Greys_r)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(target, cmap=cm.Greys_r)
ax2.set_axis_off()
ax2.set_title('target')
ax2.add_patch(plt.Rectangle((y,x), tw, th, edgecolor='w', facecolor='none', linewidth=2.5))

ax3.imshow(score_map, cmap=cm.Greys_r)
ax3.set_axis_off()
ax3.set_title('score_map')
ax3.add_patch(plt.Rectangle((y-th/2,x-tw/2), tw, th, edgecolor='w', facecolor='none', linewidth=2.5))

plt.show()
