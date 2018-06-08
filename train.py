import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from discriminator import Discriminator
from generator import Generator
from data import GaussianDataSet
from torch.utils.data import DataLoader

# parameters
# data
data_mean = 4
data_stddev = 1.25
# discriminator
d_input_size = 1
d_hidden_size = 64
d_output_size = 1
d_learning_rate = 0.0005
# generator
g_input_size = 4
g_hidden_size = 32
g_output_size = 1
g_learning_rate = 0.0005
# discriminator model
D = Discriminator(d_input_size, d_hidden_size, d_output_size)
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
# generator model
G = Generator(g_input_size, g_hidden_size, g_output_size)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)
# training
criterion = nn.BCELoss()
epochs = 10000
batch_size = 100
dataset_size = 1000
epoch_bar = 10
sine_dataset = GaussianDataSet(dataset_size, data_mean, data_stddev)
sine_dataloader = DataLoader(sine_dataset, batch_size=batch_size)
# loss
d_losses = []
g_losses = []
for epoch in range(epochs):
    D = D.train()
    G = G.train()
    d_loss = 0
    g_loss = 0
    for i, data_batch in enumerate(sine_dataloader):
        # train discriminator on real data
        D.zero_grad()
        d_real_output = D(data_batch)
        d_real_error = criterion(d_real_output, torch.ones(batch_size, 1))
        d_real_error.backward()
        # train discriminator on generated data
        g_input = torch.rand(batch_size, g_input_size)
        g_output = G(g_input).detach()
        d_gen_output = D(g_output)
        d_gen_error = criterion(d_gen_output, torch.zeros(batch_size, 1))
        d_gen_error.backward()
        d_optimizer.step()
        # calculate loss
        d_loss += float(d_real_error)
        d_loss += float(d_gen_error)
    for _ in enumerate(sine_dataloader):
        # train generator
        G.zero_grad()
        g_input = torch.rand(batch_size, g_input_size)
        g_output = G(g_input)
        d_output = D(g_output)
        g_error = criterion(d_output, 0.9*torch.ones(batch_size, 1))
        g_error.backward()
        g_optimizer.step()
        # calculate loss
        g_loss += float(g_error)
    print("epoch: ", str(epoch))
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    if epoch > epoch_bar and epoch % epoch_bar == 0:
        sine_dataset = GaussianDataSet(dataset_size, data_mean, data_stddev)
        sine_dataloader = DataLoader(sine_dataset, batch_size=batch_size)


# plot all loss
plt.figure(0)
plt.title("discriminator loss")
plt.plot(d_losses)
plt.figure(1)
plt.title("generator loss")
plt.plot(g_losses)
plt.show()
# save models
torch.save(D, 'saved_models/discriminator.pt')
torch.save(G, 'saved_models/generator.pt')



