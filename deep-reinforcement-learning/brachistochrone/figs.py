import numpy as np
import matplotlib.pyplot as plt
import sys

path = sys.argv[1]
optimal_time = 14.24

r = np.load(path+"/train.npy")
test_r = np.load(path+"/test.npy")

times = np.load(path+"/times.npy")
test_times = np.load(path+"/test_times.npy")

mu_loss = np.load(path+"/mu_losses.npy")
q_loss = np.load(path+"/q_losses.npy")

plt.plot(r, label="train rewards")
plt.plot(test_r,label="test rewards")
plt.legend()
plt.savefig(path+"/rewards.png")

plt.clf()
times = filter(times)
test_times = filter(test_times) 
plt.plot(times, label="train timings")
plt.plot(test_times,label="test timings")
plt.ylim(bottom=0)
plt.hlines(y=optimal_time, xmin=0, xmax=len(times), label="optimal time")
plt.legend(loc="upper right")
plt.savefig(path+"/times.png")

plt.clf()
plt.plot(q_loss,label="critic losses",alpha=1)
plt.legend()
plt.savefig(path+"/losses.png")

plt.clf()
plt.plot(mu_loss, label="actor losses",alpha=1)
plt.legend()
plt.savefig(path+"/losses2.png")

