import torch
import sys
from Model_Base_A_Train_DataLoad_and_Transform import get_loaders
from Model_Base_A_Test_DataLoad_and_Transform import get_test_loader
from Model_Base_A_Training import train_model
from Model_Base_A_Test_Evaluate import evaluate
from Model_Base_A import ResNet18, ResNet34
import torchsummary
from torch import nn, optim
from torch.optim import lr_scheduler
import os
from datetime import datetime

#######################################
training_epochs = 50
schedule_steps = 6
learning_rate = 0.001
batch_size = 128
num_workers = 4
dataset_name = "Galaxy10"
dataset_dir = f"/data/a2018101819/repos/실전기계학습/final_project/{dataset_name}"

model_ft = ResNet34(num_classes=10).cuda()
model_name = "ResNet34"

optimizer_name = "SGD"  # Options: "SGD", "Adam"
scheduler_name = "StepLR"  # Options: "StepLR"
#######################################

def main():
    # 1. 학습 관련 상수들 정의
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(current_dir, "result", current_time)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    pytorch_total_params = sum(p.numel() for p in model_ft.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    if int(pytorch_total_params) > 5000000:
        print('Your model has the number of parameters more than 5 millions..')
        sys.exit()
    model_ft = model_ft.to(device)
    torchsummary.summary(model_ft, (3, 256, 256))

    # 2. Define mappings for optimizers and schedulers
    optimizer_dict = {
        "SGD": lambda model: optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9),
        "Adam": lambda model: optim.Adam(model.parameters(), lr=learning_rate)
    }

    scheduler_dict = {
        "StepLR": lambda optimizer: lr_scheduler.StepLR(optimizer, step_size=schedule_steps, gamma=0.1),
    }

    # 3. Dynamically create optimizer and scheduler
    optimizer_ft = optimizer_dict[optimizer_name](model_ft)
    exp_lr_scheduler = scheduler_dict[scheduler_name](optimizer_ft)

    # 4. 데이터 로더 불러오기
    print("Loading data...")
    train_loader, val_loader = get_loaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    test_loader, test_file_names = get_test_loader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 5. 모델 학습
    import matplotlib.pyplot as plt

    # Plot train and validation loss
    def plot_loss_graph(trn_metadata, val_metadata, output_dir):
        trn_loss, _ = trn_metadata
        val_loss, _ = val_metadata

        epochs = range(1, len(trn_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, trn_loss, label='Train Loss', marker='o')
        plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Save the plot
        loss_plot_path = os.path.join(output_dir, 'train_val_loss.png')
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Loss plot saved to {loss_plot_path}")

    print("Starting training...")
    
    best_model, epoch_lst, trn_metadata, val_metadata, best_epoch = train_model(
        model=model_ft,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_epochs,
        device=device
    )
    plot_loss_graph(trn_metadata, val_metadata, output_dir)

    # 6. 모델 평가
    print("Evaluating the model...")
    preds = evaluate(
        model=best_model,
        criterion=nn.CrossEntropyLoss(),
        test_loader=test_loader,
        test_file_names=test_file_names,
        device=device,
        output_dir=output_dir,
        model_name=model_name,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        loss_func_name="CrossEntropyLoss",
        best_epoch=best_epoch,
        training_epochs=training_epochs,
        schedule_steps=schedule_steps,
        learning_rate=learning_rate,
        batch_size=batch_size
    )

    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
