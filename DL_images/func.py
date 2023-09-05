import os
import random

import seaborn as sns
import torch
import mlflow
from matplotlib import pyplot as plt
import numpy as np 
import torch.nn as nn 
import mlflow.pytorch
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models import resnet18, ResNet18_Weights



def imshow(img, mean=None, std=None):
    if (mean != None) and (std != None):
        img_new = img.clone()
        for i in range(img_new.shape[0]):
            img_new[i] = img_new[i] * std[i] + mean[i]
    else:
        img_new = img.clone()
    img_new = img_new.cpu().numpy()
    plt.imshow(np.transpose(img_new, (1, 2, 0)))  
    plt.show()


def weighted_avg_f1(output, labels):
    """Функция расчета weighted avg F1-меры"""
    # Преобразование списков в массивы numpy
    output = np.array(output)
    labels = np.array(labels)

    # Создание тензоров PyTorch
    output = torch.tensor(output)
    labels = torch.tensor(labels)

    predictions = torch.argmax(output, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    weighted_f1 = f1_score(labels, predictions, average='weighted')
    weighted_f1 = np.nan_to_num(weighted_f1, nan=0.0)  # Замена NaN на 0 при делении на 0

    return weighted_f1




def accuracy(output,labels):
    """Функция расчета accuracy"""
    # Преобразование списков в массивы numpy
    output = np.array(output)
    labels = np.array(labels)

    # Создание тензоров PyTorch
    output = torch.tensor(output)
    labels = torch.tensor(labels)

    predictions = torch.argmax(output,dim=1)
    correct = (predictions == labels).sum().cpu().numpy()
    return correct / len(labels)


def evaluate_model(model, data_loader):
    """Функция для логирования артефактов в MLflow"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)


    # Сохранение отчета в текстовый файл
    report = classification_report(true_labels, predicted_labels)
    output_file = "classification_report.txt"
    with open(output_file, "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")
    os.remove("classification_report.txt")
        
    # Save confusion matrix as CSV artifact
    df = pd.DataFrame(cm)
    new_columns = [f"predicted class {i}" for i in range(df.shape[1])]

    # Установка новых названий столбцов
    df.columns = new_columns

    # Добавление индексов
    df.insert(0, "real\pred", [f"real class {i}" for i in range(df.shape[0])])

    # Сохранение измененной таблицы 
    df.to_csv("confusion_matrix.csv", index=False)
    mlflow.log_artifact("confusion_matrix.csv")
    os.remove("confusion_matrix.csv")
   




def train_classifier(model_CNN, dataloader_train, dataloader_val, batch_size, 
                     name_save, start_weight=None, mlflow_tracking=True,
                     name_experiment=None, lr=1e-4, epochs=100,
                     scheduler=True, scheduler_step_size=10, dataset_name=None,
                     seed=42, std=None, mean=None, gamma=0.5, n_neurons=0):
    """Обучение классификационной сверточной сети

    Args:
        model_CNN: Класс модели pytorch

        dataloader_train: Обучающий даталоудер

        dataloader_val: Валидационный даталоудер

        batch_size: Размер одного батча

        name_save: Имя модели для сохранения в папку models

        start_weight: Если указать веса, то сеть будет в режиме fine tune. Defaults to None.

        mlflow_tracking (bool): Включение/выключение MLflow. Defaults to True.

        name_experiment:  Имя эксперимента для MLflow. Нужно при mlflow_tracking=True. Defaults to None.
        
        lr: Скорость обучения. Defaults to 1e-4.

        epochs: Число эпох обучения. Defaults to 100.

        scheduler (bool): Включение/выключение lr шедулера. Defaults to True.

        scheduler_step_size (int): Шаг шедулера при scheduler=True. Defaults to 10.

        dataset_name: Имя датасета для логирования в MLflow. Defaults to None.

        seed (int): Seed рандома. Defaults to 42.
        
        std: Значение СКО для нормализации. Логируется в MLflow. Defaults to None

        mean: Значение среднего для нормализации. Логируется в MLflow. Defaults to None

        gamma: Величина коэффициента lr шедулера 

        n_neurons: Число нейронов промежуточного fc слоя для ResNet
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if dataset_name != None:
        dataset_name = dataset_name.split('/')[-1]

    directory_save = 'models_classification'

    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
    
    if mlflow_tracking:
        print('С MLFLOW')
        if name_experiment == None:
            name_experiment = name_save
        with mlflow.start_run(run_name=name_experiment) as run:
            
            if model_CNN == 'resnet18':
                if start_weight != None:
                    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
                else: 
                    model = resnet18().to(device)
                # Переопредлим полносвязные слои:
                if n_neurons == 0:
                    fc_new =  nn.Sequential(
                            nn.Linear(512, 2))
                else:    
                    fc_new =  nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(512, n_neurons),
                            nn.ReLU(),
                            nn.Linear(n_neurons, 2))
                model.fc = fc_new
                mlflow.log_param("Model", 'ResNet18')
                
            else:
                model = model_CNN()
                model_class_name = model.__class__.__name__
                mlflow.log_param("Model", model_class_name)
                if start_weight != None:
                    model = torch.jit.load(start_weight)

            loss_func = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            if scheduler:
                gamma_val = gamma
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                            step_size=scheduler_step_size,
                                                            gamma=gamma_val)
            model = model.to(device)
            
            if mean != None and std != None:
                mlflow.log_param("Normalize", 'True')
                mlflow.log_param("Normalization mean", mean.tolist())
                mlflow.log_param("Normalization std", std.tolist())
            else:
                mlflow.log_param("Normalize", 'False')
            mlflow.log_param("Channels", 'RGB')
            if scheduler:
                mlflow.log_param("scheduler", 'On')
                mlflow.log_param("scheduler_step_size", scheduler_step_size)
                mlflow.log_param("scheduler_gamma", gamma_val)
            else:
                mlflow.log_param("scheduler", 'Off')

            mlflow.log_param("lr", lr)
            mlflow.log_param("optimizer", 'Adam')
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("loss", 'CrossEntropy')
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("seed", seed)
            if start_weight != None:
                mlflow.log_param("Fine-tuning", True)
            else:
                mlflow.log_param("Fine-tuning", False)

            itr_record = 0
            max_epoch_f1_val = 0
            EPOCHS = epochs

            for epoch in range(EPOCHS):
                # Обнуление градиентов параметров модели
                model.train()
                running_loss_train = 0
                all_outputs = []
                all_targets = []

                for itr, (inputs, targets) in enumerate(dataloader_train):
                    
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    batch_temp_size = inputs.shape[0]

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_func(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    if (itr + epoch * len(dataloader_train)) % 10 == 0:
                        mlflow.log_metric("train_batch_loss", loss/batch_temp_size, step=itr_record) 
                        itr_record += 1

                    all_outputs.extend(outputs.cpu().detach().numpy())
                    all_targets.extend(targets.cpu().numpy())

                    running_loss_train += loss / batch_temp_size


                epoch_loss_train = running_loss_train / len(dataloader_train)

                epoch_acc_train = accuracy(all_outputs, all_targets)
                epoch_f1_train = weighted_avg_f1(all_outputs, all_targets)


                # Валидация модели после каждой эпохи
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    all_outputs = []
                    all_targets = []
                    for i, (inputs, targets) in enumerate(dataloader_val):
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        batch_temp_size = inputs.shape[0]

                        outputs = model(inputs)

                        val_loss += loss_func(outputs, targets).item() / batch_temp_size

                        all_outputs.extend(outputs.cpu().detach().numpy())
                        all_targets.extend(targets.cpu().numpy())

                    # Вычисление среднего значения функции потерь на валидационном наборе данных
                    epoch_loss_val =  val_loss / len(dataloader_val)
                    epoch_acc_val = accuracy(all_outputs, all_targets)
                    epoch_f1_val = weighted_avg_f1(all_outputs, all_targets)

                    mlflow.log_metric("validation_epoch_accuracy", epoch_acc_val, step=(epoch+1))
                    mlflow.log_metric("validation_epoch_loss", epoch_loss_val, step=(epoch+1))
                    mlflow.log_metric("validation_epoch_f1", epoch_f1_val, step=(epoch+1))

                if scheduler:
                    lr_scheduler.step()
                    
                # Вывод значения функции потерь на каждой 5 эпохе
                if ((epoch+1) % 5 == 0) or epoch==0:
                    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss_train:.4f},'
                            f' Train Aсс: {epoch_acc_train:.4f}'
                            f' Val Loss: {epoch_loss_val:.4f}, Val Acc:{epoch_acc_val:.4f} ')

                mlflow.log_metric("train_epoch_accuracy", epoch_acc_train, step=(epoch+1))
                mlflow.log_metric("train_epoch_loss", epoch_loss_train, step=(epoch+1))
                mlflow.log_metric("train_epoch_f1", epoch_f1_train, step=(epoch+1))
                
                if epoch >= 3 and epoch_f1_val > max_epoch_f1_val:
                    max_epoch_f1_val = epoch_f1_val
                    acc_model = epoch_acc_val
                    model_to_save = torch.nn.Sequential(
                        model,
                        torch.nn.Softmax(1),
                    )
                    epoch_best = epoch+1
                    model_scripted = torch.jit.script(model_to_save)
                    name_save_model = directory_save + '/' + name_save +'.pt'
                    model_scripted.save(name_save_model) 
                    #torch.save(model, directory_save + '/' + name_save +'.pth')
                    evaluate_model(model=model, data_loader=dataloader_val)
                
            print('Обучение завершено')
            print(f'Сохранена модель {name_save_model} с лучшим weighted avg f1 на валидации = {max_epoch_f1_val}')  
            print('Accuracy данной модели равно', acc_model) 

            mlflow.log_metric("max f1 saved model", max_epoch_f1_val)
            mlflow.log_metric("accuracy of model", acc_model)
            mlflow.log_metric("epoch of save", epoch_best)


            mlflow.log_artifact(name_save_model)
    else:
        print('БЕЗ MLFLOW не предусмотрено')
        return










##################################################################################








def train_regressor(model_CNN, dataloader_train, dataloader_val, batch_size, 
                     name_save, start_weight=None, mlflow_tracking=True,
                     name_experiment=None, lr=1e-4, epochs=100,
                     scheduler=True, scheduler_step_size=10, dataset_name=None,
                     seed=42, std=None, mean=None, gamma=0.5, n_neurons=0):
    """Обучение классификационной сверточной сети

    Args:
        model_CNN: Класс модели pytorch

        dataloader_train: Обучающий даталоудер

        dataloader_val: Валидационный даталоудер

        batch_size: Размер одного батча

        name_save: Имя модели для сохранения в папку models

        start_weight: Если указать веса, то сеть будет в режиме fine tune. Defaults to None.

        mlflow_tracking (bool): Включение/выключение MLflow. Defaults to True.

        name_experiment:  Имя эксперимента для MLflow. Нужно при mlflow_tracking=True. Defaults to None.
        
        lr: Скорость обучения. Defaults to 1e-4.

        epochs: Число эпох обучения. Defaults to 100.

        scheduler (bool): Включение/выключение lr шедулера. Defaults to True.

        scheduler_step_size (int): Шаг шедулера при scheduler=True. Defaults to 10.

        dataset_name: Имя датасета для логирования в MLflow. Defaults to None.

        seed (int): Seed рандома. Defaults to 42.
        
        std: Значение СКО для нормализации. Логируется в MLflow. Defaults to None

        mean: Значение среднего для нормализации. Логируется в MLflow. Defaults to None

        gamma: Величина коэффициента lr шедулера 

        n_neurons: Число нейронов промежуточного fc слоя для ResNet
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    DEVICE = device 
    train_loader = dataloader_train 
    val_loader = dataloader_val
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if dataset_name != None:
        dataset_name = dataset_name.split('/')[-1]

    directory_save = 'models_regressor'

    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
    
    if mlflow_tracking:
        print('С MLFLOW')
        if name_experiment == None:
            name_experiment = name_save
        with mlflow.start_run(run_name=name_experiment) as run:
            
            if model_CNN == 'resnet18':
                if start_weight != None:
                    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
                else: 
                    model = resnet18().to(device)
                # Переопредлим полносвязные слои:
                if n_neurons == 0:
                    fc_new =  nn.Sequential(
                            nn.Linear(512, 1))
                else:    
                    fc_new =  nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(512, n_neurons),
                            nn.ReLU(),
                            nn.Linear(n_neurons, 1))
                model.fc = fc_new
                mlflow.log_param("Model", 'ResNet18')
                
            else:
                model = model_CNN()
                model_class_name = model.__class__.__name__
                mlflow.log_param("Model", model_class_name)
                if start_weight != None:
                    model = torch.jit.load(start_weight)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            if scheduler:
                gamma_val = gamma
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                            step_size=scheduler_step_size,
                                                            gamma=gamma_val)
            model = model.to(device)
            
            if mean != None and std != None:
                mlflow.log_param("Normalize", 'True')
                mlflow.log_param("Normalization mean", mean.tolist())
                mlflow.log_param("Normalization std", std.tolist())
            else:
                mlflow.log_param("Normalize", 'False')
            mlflow.log_param("Channels", 'RGB')
            if scheduler:
                mlflow.log_param("scheduler", 'On')
                mlflow.log_param("scheduler_step_size", scheduler_step_size)
                mlflow.log_param("scheduler_gamma", gamma_val)
            else:
                mlflow.log_param("scheduler", 'Off')

            mlflow.log_param("lr", lr)
            mlflow.log_param("optimizer", 'Adam')
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("loss", 'MSE')
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("seed", seed)
            if start_weight != None:
                mlflow.log_param("Fine-tuning", True)
            else:
                mlflow.log_param("Fine-tuning", False)

            N_EPOCHS  = epochs
            name_model = name_save
            best_mse = np.inf

            for epoch in range(N_EPOCHS):
                # Обнуление градиентов параметров модели
                model.train()
                batches_train = 0
                batches_val = 0
                running_loss_train = 0
                running_corrects_train_EF_5 = 0
                running_corrects_train_EF_10 = 0
                running_corrects_val_EF_5 = 0
                running_corrects_val_EF_10 = 0
                target_EF_all = []
                output_EF_all = []

                for inputs, targets in train_loader:
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)
                    batch_temp_size = inputs.shape[0]

                    # Обнуление градиентов параметров модели
                    optimizer.zero_grad()

                    # Прямой проход (forward pass)
                    outputs = model(inputs).to(torch.float64)

                    # Вычисление значения функции потерь
                    loss = criterion(outputs, targets)
                    running_loss_train += loss.item()

                    # Обратное распространение ошибки (backward pass)
                    loss.backward()

                    # Обновление параметров модели
                    optimizer.step()

                    output_EF = outputs[:, 0]
                    target_EF = targets[:, 0]

                    
                    running_corrects_train_EF_5 += torch.sum(torch.abs(output_EF - target_EF) < 5)
                    running_corrects_train_EF_10 += torch.sum(torch.abs(output_EF - target_EF) < 10)


                    batches_train += batch_temp_size

                    # Добавляем значения в списки
                    output_EF_all.extend(output_EF.cpu().detach().numpy())
                    target_EF_all.extend(target_EF.cpu().numpy())

                # Преобразовываем списки в тензоры PyTorch
                output_EF_all = torch.tensor(output_EF_all)
                target_EF_all = torch.tensor(target_EF_all)

                # Вычисляем метрики для всего набора данных на эпохе
                mse_epoch_train = torch.nn.functional.mse_loss(output_EF_all, target_EF_all)
                mae_epoch_train = torch.nn.functional.l1_loss(output_EF_all, target_EF_all)

                epoch_acc_train_EF_5 = running_corrects_train_EF_5 / batches_train
                epoch_acc_train_EF_10 = running_corrects_train_EF_10 / batches_train

                mlflow.log_metric("train_epoch_MSE", mse_epoch_train, step=(epoch+1))
                mlflow.log_metric("train_epoch_MAE", mae_epoch_train, step=(epoch+1))

                mlflow.log_metric("train_epoch_accuracy_EF_10", epoch_acc_train_EF_10, step=(epoch+1))
                mlflow.log_metric("train_epoch_accuracy_EF_5", epoch_acc_train_EF_5, step=(epoch+1))     
   

                # Валидация модели после каждой эпохи
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    output_EF_all = []
                    target_EF_all = []
                    for i, (inputs, targets) in enumerate(val_loader):
                        inputs = inputs.to(DEVICE)
                        targets = targets.to(DEVICE)
                        batch_temp_size = inputs.shape[0]

                        # Прямой проход (forward pass)
                        outputs = model(inputs)

                        # Вычисление значения функции потерь
                        val_loss += criterion(outputs, targets).item()

                        output_EF = outputs[:, 0]
                        target_EF = targets[:, 0]
                        
                        running_corrects_val_EF_5 += torch.sum(torch.abs(output_EF - target_EF) < 5)
                        running_corrects_val_EF_10 += torch.sum(torch.abs(output_EF - target_EF) < 10)


                        batches_val += batch_temp_size

                        # Добавляем значения в списки
                        output_EF_all.extend(output_EF.cpu().detach().numpy())
                        target_EF_all.extend(target_EF.cpu().numpy())

                    # Преобразовываем списки в тензоры PyTorch
                    output_EF_all = torch.tensor(output_EF_all)
                    target_EF_all = torch.tensor(target_EF_all)

                    # Вычисляем метрики для всего набора данных на эпохе
                    mse_epoch_val = torch.nn.functional.mse_loss(output_EF_all, target_EF_all)
                    mae_epoch_val = torch.nn.functional.l1_loss(output_EF_all, target_EF_all)

                    epoch_acc_val_EF_5 = running_corrects_val_EF_5 / batches_val
                    epoch_acc_val_EF_10 = running_corrects_val_EF_10 / batches_val

                    mlflow.log_metric("val_epoch_MSE", mse_epoch_val, step=(epoch+1))
                    mlflow.log_metric("val_epoch_MAE", mae_epoch_val, step=(epoch+1))

                    mlflow.log_metric("val_epoch_accuracy_EF_10", epoch_acc_val_EF_10, step=(epoch+1))
                    mlflow.log_metric("val_epoch_accuracy_EF_5", epoch_acc_val_EF_5, step=(epoch+1))     

                if scheduler:
                    lr_scheduler.step()

                # Вывод значения функции потерь на каждой 10 эпохе
                if ((epoch+1) % 5 == 0) or epoch==0:
                    print(f'Epoch {epoch+1}/{N_EPOCHS}, Train MSE: {mse_epoch_train:.3f}, Train MAE: '
                    f'{mae_epoch_train:.3f} Val MSE: {mse_epoch_val:.3f}, Val MAE:{mae_epoch_val:.3f}')
                
                if epoch > 5 and mse_epoch_val < best_mse:
                    best_mse = mse_epoch_val
                    mae_saved = mae_epoch_val
                    model_scripted = torch.jit.script(model)
                    name_save = directory_save + '/' + name_model +'.pt'
                    model_scripted.save(name_save) 
                
            print('Обучение завершено')
            print(f'Сохранена модель {name_save} с лучшим MSE на валидации = {best_mse}')   
            print(f'У данной модели MAE на валидации = {mae_saved}') 
            mlflow.log_metric("MSE saved model", best_mse)
            mlflow.log_metric("MAE saved model", mae_saved)
            mlflow.log_artifact(name_save)

            # Логирование гистограмм:
            model = torch.jit.load(name_save).to(DEVICE)
            delta_EF = []
            with torch.no_grad():
                val_loss = 0.0
                for inputs, targets in val_loader:
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)
                    outputs = model(inputs)

                    output_EF = outputs[:, 0]
                    target_EF = targets[:, 0]

                    delta_EF.extend((target_EF-output_EF).tolist())
            # Создание графика EF
            sns.set(style="whitegrid")
            plt.figure(figsize=(7, 5))
            sns.histplot(x=np.round(np.array(delta_EF)), kde=True, bins=30)
            plt.xlabel("Delta percent")
            plt.title("EF")
            plt.savefig("hist_val_EF.png")
            plt.close()

            # Загрузка изображений в MLflow
            mlflow.log_artifact("hist_val_EF.png")
            os.remove("hist_val_EF.png")

    else:
        print('БЕЗ MLFLOW не предусмотрено')
        return