# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm, t

# fig, ax = plt.subplots()
# xlim = [-15, 15]
# mu = 0
# s = np.arange(1, 6)
# x = np.linspace(xlim[0], xlim[1], 1000)
# y = norm.pdf(x.reshape(-1, 1), mu, s)
# # print(y.shape)
# label = ["$\sigma$={}".format(i) for i in s]
# plt.plot(x, y, label=label)
# plt.legend()
# plt.show()
# import multiprocessing
from playsound import playsound

sound_file = "../multimedia_py39/taylor_music.wav"
# p = multiprocessing.Process(target=playsound, args=(sound_file,))
# p.start()
playsound(sound_file)
# input("press ENTER to stop playback")
# p.terminate()