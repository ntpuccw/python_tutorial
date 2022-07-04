import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
x = np.linspace(-3*np.pi, 3*np.pi, 200)
y = np.sin(x)
line, = ax.plot(x, y)
ax.set_ylim([-2.5, 2.5])
num_terms = 30
def animate(i):
    y = 0
    for j in range(i):
        y += np.sin((2*j+1)*x)/(2*j+1)
    
    line.set_ydata(y)
    return line,


ani = animation.FuncAnimation(
    fig, animate, frames= np.arange(1, num_terms), interval=100, 
        blit=True, save_count=50, repeat=False)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()