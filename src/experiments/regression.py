import sys
sys.path.append(".")
sys.path.append("src")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import time

from src.networks.simple_nn import simpleBayesian, simpleNN

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(5)

# Train data (is just an N shaped function)
x_train = torch.linspace(-3, 3, 500)
print(x_train[:5])
y_train = x_train.pow(3) - x_train.pow(2) + 5*torch.rand(x_train.size())
x_train = torch.unsqueeze(x_train, dim=1).to(device)
y_train = torch.unsqueeze(y_train, dim=1).to(device)

# Eval data
x_eval = torch.linspace(-3, 3, 100)
y_eval = x_eval.pow(3) - x_eval.pow(2) + 5*torch.rand(x_eval.size())
x_eval = torch.unsqueeze(x_eval, dim=1).to(device)
y_eval = torch.unsqueeze(y_eval, dim=1).to(device)

# test data
x_test = torch.linspace(-3, 3, 500)
y_test = x_test.pow(3) - x_test.pow(2) + 5*torch.rand(x_test.size())
x_test = torch.unsqueeze(x_test, dim=1).to(device)
y_test = torch.unsqueeze(y_test, dim=1).to(device)

# define model, loss & optimizer
# model = simpleBayesian(input_dim=1, hid_dim=100, output_dim=1, inference_reps=5, device=device)
model = simpleNN(input_dim=1, hid_dim=100, output_dim=1, device=device)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 100_000
best_eval_loss = 10_000  # temp bad evaluation
best_eval_epoch = 0
train_l, eval_l = [], []
best_state_dict = model.state_dict()
model.train()

# train model
start = time.time()
for step in range(epochs):
    # train step
    train_preds = model(x_train)
    train_loss = loss(train_preds, y_train)
    train_l.append(train_loss.cpu().detach().numpy())

    # backpropagation step
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # evaluation step
    eval_preds = model(x_eval)
    eval_loss = loss(eval_preds, y_eval)
    eval_l.append(eval_loss.cpu().detach().numpy())

    # save weights
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        best_eval_epoch = step
        best_state_dict = model.state_dict()

    if step % 1000 == 0:
        print(f"ep: {step}, train loss: {train_loss:.2f}, eval_loss: {eval_loss:.2f}")

end = time.time()
print(f"Trained for {((end-start)/60):.2f} minutes.")

# save train plot
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.plot(
    range(epochs),
    train_l,
    color="r",
    linewidth=2,
    label="train",
)
plt.plot(
    range(epochs),
    eval_l,
    color="b",
    linewidth=2,
    label="eval",
)
plt.title(f"{model.name}")
plt.ylim(0, 40)
plt.legend()
plt.savefig(f"results/{model.name}_training.png")

# make predictions
print(f"Loading model weights of epoch: {best_eval_epoch}")
model.load_state_dict(best_state_dict)
repeats = 10
y_mean = torch.stack([model(x_test) for i in range(repeats)]).mean(dim=0)
print(f"\nPrediction loss: {loss(y_mean, y_test)}")

# create fitness plot
plt.clf()
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.scatter(
    x_train.data.cpu().numpy(),
    y_train.data.cpu().numpy(),
    color="cyan",
    s=4,
    label="train data",
)
plt.scatter(
    x_test.data.cpu().numpy(),
    y_test.data.cpu().numpy(),
    color="black",
    s=2,
    label="test data",
) 
plt.plot(
    x_test.data.cpu().numpy(),
    y_mean.data.cpu().numpy(),
    color="r",
    linewidth=2,
    label="predictions",
)
plt.title(f"{model.name}, train loss: {train_loss:.2f} test loss: {loss(y_mean, y_test):.2f}")
plt.legend()
plt.savefig(f"results/{model.name}_pred.png")