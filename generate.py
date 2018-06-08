import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
g_input_size = 4
data_size = 10000
G = torch.load('saved_models/generator.pt')
G = G.eval()
g_input = torch.rand(data_size, g_input_size)
g_output = G(g_input)
data = []
for i in range(data_size):
    data.append(float(g_output[i][0]))
sns.distplot(data)
print("mean:", str(np.mean(data)))
print("stddev", str(np.std(data)))
plt.show()