import matplotlib.pyplot as plt
import pickle
with open('temp_files/counter_plot', 'rb') as fp:
	a = pickle.load(fp)
plt.plot(a)
plt.savefig('temp_files/train_curve.png')
plt.show()
