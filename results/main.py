import numpy as np
import matplotlib.pyplot as plt

avg_accuracy = np.loadtxt('avg_accuracy.txt')
print(avg_accuracy.shape)
error_and_size = np.loadtxt('error_accuracy_circuit-size.txt')


average_accuracy = np.mean(avg_accuracy, axis=1)
average_accuracy = average_accuracy[::-1]
std_accuracy = np.std(avg_accuracy, axis=1)
error = error_and_size[:,0]
ACC = error_and_size[:,1]
size = error_and_size[:,2]
# print('error', error)
# print('size', size)
# print('ACC', ACC)
# print(average_accuracy)



cor_rgb1 = (128 / 255, 0 / 255, 255 / 255)
cor_rgb2 = (69 / 255, 67 / 255, 68 / 255)

plt.rcParams["figure.figsize"] = (7.2, 4.0)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})

plt.plot(error, average_accuracy, color=cor_rgb2)
plt.errorbar(
    error, average_accuracy, yerr=std_accuracy, 
    fmt='o', 
    label='Data with Standard Deviation',
    color=cor_rgb1
    )
# plt.grid(True)
# plt.title('average accuracy for approximated quantum circuit')
plt.xlabel('error')
plt.ylabel('accuracy')
plt.savefig('curva_media.svg', dpi=300)
plt.show()