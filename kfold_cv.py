import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def kfold_cv(dataset, model, trainloader, validloader, optimizer, criterion, num_folds):
    for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
        # Create new instances of the model for each fold
        model = DepthNet(4, use_transpose_conv=True)
        optimizer = Adam(model.parameters(), lr=1e-3)

        # Create new instances of data loaders for each fold
        trainloader, validloader, testloader = utils.get_train_valid_test_loaders(dataset, batchsize=32, split=(0.8, 0.1, 0.1))

        # Train the model
        train(model, trainloader, validloader, optimizer, criterion)

        # Validate the model
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for inputs, labels in validloader:
                outputs = model(inputs)
                y_true.extend(labels.numpy())
                y_pred.extend(outputs.numpy())

        # Calculate validation loss
        validation_loss = mean_squared_error(y_true, y_pred)
        all_validation_losses.append(validation_loss)

        # Calculate accuracy
        accuracy = your_accuracy_function(y_true, y_pred)  # Replace with your actual accuracy function
        all_accuracies.append(accuracy)

        # Print and save metrics
        print(f"Fold {fold}: Validation Loss: {validation_loss}, Accuracy: {accuracy}")

    return all_validation_losses, all_accuracies