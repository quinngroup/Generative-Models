import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
mnist = np.load("../data/mnist_test_seq.npy")
print(mnist.shape)

test = mnist[:,0]
print(test.shape)

fig = plt.figure()
im = plt.imshow(test[0], interpolation='none')
ax = plt.axes(xlim=(0,64), ylim=(0,64))
line, = ax.plot([], [], lw=2)

def init():
    im.set_data(test[0])
    return [im]
    
def animate(i):
    im.set_array(test[i])
    return [im]
    
anim = ani.FuncAnimation(fig, animate, init_func=init, frames=20, interval=20, blit=True)

plt.show()