import numpy as np
import matplotlib.pyplot as plt

results = np.loadtxt("error_accuracy_circuit-size.txt")
error = results[:, 0]
accuracy = np.loadtxt("avg_accuracy.txt").T
accuracy = np.mean(accuracy, axis=1)
print(accuracy)

circuit_size = results[:, 2]

# plt.plot(np.log(error), np.log(accuracy))
# plt.plot(np.log(error), np.log(circuit_size))
plt.plot(error, accuracy)
plt.show()
