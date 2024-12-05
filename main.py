import torch
import sys
from Model_Base_A_Train_DataLoad_and_Transform import get_loaders
from Model_Base_A_Test_DataLoad_and_Transform import get_test_loader
from Model_Base_A_Training import train_model
from Model_Base_A_Test_Evaluate import evaluate
from Model_Base_A import ResNet18, ResNet34, ResNet152, ResNet50
import torchsummary
from torch import nn, optim
from torch.optim import lr_scheduler
import os
from datetime import datetime
import torch.nn.functional as F
import time

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:  
            # 기본값으로 모든 클래스에 동일한 가중치
            self.alpha = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 
                                       1.0, 1.0, 1.0, 1.0, 1.0,])
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # self.alpha를 GPU로 이동
        if inputs.is_cuda:  
            self.alpha = self.alpha.to(inputs.device)

        # inputs의 크기: [batch_size, num_classes]
        # targets의 크기: [batch_size]
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        alpha_factor = self.alpha[targets]
        pt = torch.exp(-CE_loss)  # 각 타겟에 대해 해당 클래스의 alpha 값
        F_loss = alpha_factor * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def main():
    since = time.time()
    # 1. 학습 관련 상수들 정의
    #######################################
    training_epochs = 20
    schedule_steps = 6
    learning_rate = 0.0001
    batch_size = 64
    num_workers = 4
    dataset_dir = "/data/a2018101819/repos/실전기계학습/final_project/Galaxy10"
    
    model_ft = ResNet18(num_classes=10).cuda()
    model_name = "ResNet18"
    #######################################

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

    # 3. Loss, Optimizer, Scheduler 설정
    #######################################
    loss_func = FocalLoss(alpha=[0.866551127, 5.319148936, 1.642036125, 0.946969697, 1.25,
                                 0.875656743, 0.956937799, 0.670241287, 0.675219446, 0.968054211
                                 ], gamma=1)
    loss_func_name = f"{loss_func}"
    criterion = loss_func
    
    optimizer = optim.Adam(model_ft.parameters(), lr=learning_rate)
    optimizer_name = f"{optimizer}"
    optimizer_ft = optimizer
    
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=schedule_steps, gamma=0.1)
    scheduler_name = f"{scheduler}"
    exp_lr_scheduler = scheduler
    
    #######################################
    
    # 4. 데이터 로더 불러오기
    print("Loading data...")
    train_loader, val_loader = get_loaders(
        dataset_dir =dataset_dir ,
        batch_size =batch_size , 
        num_workers =num_workers 
        )
    test_loader,test_file_names = get_test_loader(
        dataset_dir = dataset_dir,
        batch_size = batch_size, 
        num_workers = num_workers
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
    
    best_model, epoch_lst, trn_metadata, val_metadata ,best_epoch= train_model(
        model=model_ft,
        criterion=criterion,
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
    preds = evaluate(model = best_model, 
                     criterion= criterion, 
                      test_loader = test_loader, 
                      test_file_names = test_file_names,
                      device= device, 
                      output_dir= output_dir, 
                      model_name = model_name,
                      optimizer_name=optimizer_name, 
                      scheduler_name=scheduler_name, 
                      loss_func_name = loss_func_name,
                      best_epoch=best_epoch,
                      training_epochs= training_epochs,
                      schedule_steps = schedule_steps,
                      learning_rate = learning_rate,
                      batch_size =batch_size
                      )

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("All tasks completed successfully.")
    
    

if __name__ == "__main__":
    main()

## 스케줄러, 로스펑션, 옵티마이저 이름 문자열로 넘기기
