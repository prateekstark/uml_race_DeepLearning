import matplotlib.pyplot as plt
import pickle
with open('temp_files_MCTS/counter_plot', 'rb') as fp:
	a = pickle.load(fp)
plt.plot(a)
plt.savefig('temp_files_MCTS/train_curve.png')
plt.show()
