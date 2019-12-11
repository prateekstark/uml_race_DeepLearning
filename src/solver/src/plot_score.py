import matplotlib.pyplot as plt
import pickle
with open('counter_plot', 'rb') as fp:
	a = pickle.load(fp)
plt.plot(a)
plt.savefig('train_curve.png')
plt.show()
