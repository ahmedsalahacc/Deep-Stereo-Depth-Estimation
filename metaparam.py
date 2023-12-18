import matplotlib.pyplot as plt
from torch.optim import Adam, SGD, RMSprop
from trainer import train
import utils

def test_batch_sizes(inputs, dataset, batch_sizes=[16, 32, 64]):
  all_training_losses_batch = []
  all_validation_losses_batch = []

  for batch_size in batch_sizes:
    trainloader, validloader, testloader = utils.get_train_valid_test_loaders(dataset, batchsize=batch_size, split=(0.8, 0.1, 0.1))
    inputs["trainloader"] = trainloader
    inputs["validloader"] = validloader
    inputs["training_losses"] = [0]*30
    inputs["validation_losses"] = [0]*30
    train(**inputs)
    all_training_losses_batch.append(inputs["training_losses"])
    all_validation_losses_batch.append(inputs["validation_losses"])

  for loss in all_validation_losses_batch:
    plt.plot(loss[2:])
    plt.xlabel('Number of epochs')
    plt.ylabel('Validation loss')
    plt.legend(['16', '32', '64'])
    plt.title('Validation losses against different batch sizes')
    plt.savefig("exp_batch_og.png", dpi=1000)

def test_optimizers(inputs, optimizers):
  all_training_losses_optim = []
  all_validation_losses_optim = []

  for optimizer in optimizers:
    inputs["optimizer"] = optimizer
    inputs["training_losses"] = [0]*inputs["epochs"]
    inputs["validation_losses"] = [0]*inputs["epochs"]
    train(**inputs)
    all_training_losses_optim.append(inputs["training_losses"])
    all_validation_losses_optim.append(inputs["validation_losses"])

  for loss in all_validation_losses_optim:
    plt.plot(loss[1:])
    plt.xlabel('Number of epochs')
    plt.ylabel('Validation loss')
    plt.legend(['Adam', 'SGD', 'RMSprop'])
    plt.title('Validation losses against different optimizers')
    plt.savefig("exp_optim.png", dpi=1000)
