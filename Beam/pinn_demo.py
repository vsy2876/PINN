import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

def exact_solution(l, M, x):
    "Defines the analytical solution to the under-damped harmonic oscillator problem above."
    E = 2 * 10 ** 11
    I = 50 * 1e-9
    u = (M/(2*E*I)) * x.pow(2)
    return u

class FCN(nn.Module):
    "Defines a fully-connected network in PyTorch"
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

# define a neural network to train
pinn = FCN(1,1,64,3)

# define boundary points, for the boundary loss
x_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)# (1, 1)

# define training points over the entire domain, for the physics loss
x_physics = torch.linspace(0,2,20).view(-1,1).requires_grad_(True)# (20, 1)

# train the PINN
l, M = 2, -300
E = 2 * 10 ** 11
I = 50 * 1e-9
x_test = torch.linspace(0,2,200).view(-1,1) #(200, 1)
u_exact = exact_solution(l, M, x_test)
optimiser = torch.optim.Adam(pinn.parameters(),lr=1e-4)
num_epochs = 800
losses = []
for i in range(num_epochs+1):
    optimiser.zero_grad()

    # compute each term of the PINN loss function above
    # using the following hyperparameters
    lambda1, lambda2 = 0.7, 0.3

    # compute boundary loss
    u = pinn(x_boundary)# (1, 1)
    loss1 = (torch.squeeze(u) - 0)**2
    dudx = torch.autograd.grad(u, x_boundary, torch.ones_like(u), create_graph=True)[0]# (1, 1)
    loss2 = (torch.squeeze(dudx) - 0)**2

    # compute physics loss
    u = pinn(x_physics)# (20, 1)
    dudx = torch.autograd.grad(u, x_physics, torch.ones_like(u), create_graph=True)[0]# (20, 1)
    d2udx2 = torch.autograd.grad(dudx, x_physics, torch.ones_like(dudx), create_graph=True)[0]# (20, 1)
    loss3 = torch.mean((d2udx2 - M/(E*I))**2)


    # backpropagate joint loss, take optimiser step
    loss = loss1 + lambda1 * loss2 + lambda2 * loss3
    loss.backward()
    optimiser.step()
    losses.append(loss.item())

    # plot the result as training progresses
    if i % 200 == 0:
        #print(u.abs().mean().item(), dudt.abs().mean().item(), d2udt2.abs().mean().item())
        u = pinn(x_test).detach()
        plt.figure(figsize=(6,2.5))
        plt.scatter(x_physics.detach()[:,0], torch.zeros_like(x_physics)[:,0], s=20, lw=0, color="tab:green", alpha=0.6)
        plt.scatter(x_boundary.detach()[:,0], torch.zeros_like(x_boundary)[:,0], s=20, lw=0, color="tab:red", alpha=0.6)
        plt.plot(x_test[:,0], u_exact[:,0], label="Exact solution", color="tab:grey", alpha=0.6)
        plt.plot(x_test[:,0], u[:,0], label="PINN solution", color="tab:green")
        plt.title(f"Training step {i}")
        plt.legend()
        plt.show()
        print(f"Epoch [{i+1}/{num_epochs}], Loss: {loss.item():.4f}")
    # plt.close()



plt.plot(range(1, num_epochs+2), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()