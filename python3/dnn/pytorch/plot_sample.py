# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.plot(1, 1, marker='.')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 10

plt.figure(figsize=(8,3))

ax = plt.subplot(121)
x = np.arange(0,10,0.001)
ax.plot(x, np.sin(np.sinc(x)), 'r', lw=2)
ax.set_title('Nice wiggle')

plt.show()


