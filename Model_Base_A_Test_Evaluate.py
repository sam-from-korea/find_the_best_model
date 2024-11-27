import itertools
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
def evaluate(model, criterion, test_loader,test_file_names, device, output_dir,model_name = None, optimizer_name=None, scheduler_name=None, loss_func_name = None,best_epoch=None,training_epochs = None,
             schedule_steps = None,
                learning_rate = None,
                batch_size = None, ):
    """
    Evaluate the model on the test dataset and save predictions, model information, and weights.

    Args:
        model: The trained model to evaluate.
        criterion: Loss function used during evaluation.
        test_loader: DataLoader for the test dataset.
        device: The device to use ('cuda' or 'cpu').
        output_dir: Directory to save outputs (default: '/content/').
        optimizer: Optimizer used during training.
        scheduler: Learning rate scheduler used during training.
        best_epoch: The epoch at which the best model was saved.
        split_ratios: Dataset train/test/val split ratios (e.g., (0.7, 0.2, 0.1)).
        random_state: The random state/seed used for dataset splitting.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set the model to evaluation mode
    model.eval()
    preds = []
    gts = []  # Ground truth
    file_name_mapping = []
# Calculate test accuracy
    correct = 0
    total = 0
    for batch_idx , (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + inputs.size(0)
            batch_file_names = test_file_names[start_idx:end_idx]

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            preds.append(predicted)
            gts.append(labels)
            file_name_mapping.extend(batch_file_names)
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")
    # Process predictions
    category = [t.cpu().numpy() for t in preds]
    t_category = list(itertools.chain(*category))
    actual = [t.cpu().numpy() for t in gts]
    t_actual = list(itertools.chain(*actual))
    Id = list(range(0, len(t_category)))

# Combine into a dictionary
    prediction = {
        'Id': Id,
        'Actual': t_actual,  # Add actual values
        'Predicted': t_category , # Predicted values
        "File Name": file_name_mapping,
    }

    # Generate confusion matrix
    conf_matrix = confusion_matrix(t_actual, t_category)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=range(10), columns=range(10))

    # Save confusion matrix as CSV
    conf_matrix_csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
    conf_matrix_df.to_csv(conf_matrix_csv_path)
    print(f"Confusion Matrix saved to {conf_matrix_csv_path}")

    # Plot confusion matrix and save as image
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    conf_matrix_img_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(conf_matrix_img_path)
    plt.close()
    print(f"Confusion Matrix plot saved to {conf_matrix_img_path}")

    # Save predictions to CSV
    prediction_csv_path = os.path.join(output_dir, 'prediction.csv')
    prediction_df = pd.DataFrame(prediction,columns=['Id', 'Actual', 'Predicted',"File Name"])
    prediction_df.to_csv(prediction_csv_path, index=False)

    print(f"Predictions saved to {prediction_csv_path}")

    # Save model weights to .pth file
    model_weights_path = os.path.join(output_dir, 'model_weights.pth')
    torch.save(model.state_dict(), model_weights_path)
    print(f"Model weights saved to {model_weights_path}")

    # Save detailed model information to a text file
    model_info_path = os.path.join(output_dir, 'model_info.txt')
    with open(model_info_path, 'w') as f:
        f.write("### Model Information ###\n")
        f.write(f"Device: {device}\n")
        f.write(f"ModelName: {model_name}\n")
        f.write(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}\n\n")

        f.write("### Optimizer Information ###\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f"Scheduler: {scheduler_name}\n")
        
        f.write("### Loss Function ###\n")
        f.write(f"Loss Function: {loss_func_name}\n\n")

        f.write("### Training Settings ###\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Scheduler Step Size: {schedule_steps}\n")
        f.write(f"Total Training Epochs: {training_epochs}\n")
        f.write(f"Best Model Saved at Epoch: {best_epoch}\n")

        f.write("\n### Prediction Results ###\n")
        f.write(f"Total Predictions: {len(t_category)}\n")
        f.write(f"Prediction CSV Path: {prediction_csv_path}\n")
        f.write(f"Model Weights Path: {model_weights_path}\n")

        f.write(f"### Test Results ###\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    print(f"Model information saved to {model_info_path}")

    return preds
