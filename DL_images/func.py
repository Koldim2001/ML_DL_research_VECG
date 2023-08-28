import os
import random

import torch
import mlflow
from matplotlib import pyplot as plt
import numpy as np 
import torch.nn as nn 
import mlflow.pytorch
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report


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

    predictions = torch.argmax(output, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    weighted_f1 = f1_score(labels, predictions, average='weighted')
    weighted_f1 = np.nan_to_num(weighted_f1, nan=0.0)  # Замена NaN на 0 при делении на 0

    return weighted_f1




def accuracy(output,labels):
    """Функция расчета accuracy"""

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
                     seed=42, std=None, mean=None, gamma=0.5):
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

    directory_save = 'models'

    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
    
    if mlflow_tracking:
        print('С MLFLOW')
        if name_experiment == None:
            name_experiment = name_save
        with mlflow.start_run(run_name=name_experiment) as run:
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
            max_epoch_acc_val = 0
            EPOCHS = epochs

            for epoch in range(EPOCHS):
                # Обнуление градиентов параметров модели
                model.train()
                running_loss_train = 0
                running_corrects_train = 0
                running_f1_train = 0
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

                    running_loss_train += loss / batch_temp_size
                    running_corrects_train += accuracy(outputs, targets)
                    running_f1_train += weighted_avg_f1(outputs, targets)


                epoch_loss_train = running_loss_train / len(dataloader_train)
                epoch_acc_train = running_corrects_train / len(dataloader_train)
                epoch_f1_train = running_f1_train / len(dataloader_train)

                # Валидация модели после каждой эпохи
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    running_corrects_val = 0
                    running_f1_val = 0
                    for i, (inputs, targets) in enumerate(dataloader_val):
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        batch_temp_size = inputs.shape[0]

                        outputs = model(inputs)

                        val_loss += loss_func(outputs, targets).item() / batch_temp_size
                        running_corrects_val += accuracy(outputs, targets)  
                        running_f1_val += weighted_avg_f1(outputs, targets)

                    # Вычисление среднего значения функции потерь на валидационном наборе данных
                    epoch_loss_val =  val_loss / len(dataloader_val)
                    epoch_acc_val =  running_corrects_val / len(dataloader_val)
                    epoch_f1_val =  running_f1_val / len(dataloader_val)

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

                    model_scripted = torch.jit.script(model_to_save)
                    name_save_model = directory_save + '/' + name_save +'.pt'
                    model_scripted.save(name_save_model) 
                    evaluate_model(model=model, data_loader=dataloader_val)
                
            print('Обучение завершено')
            print(f'Сохранена модель {name_save_model} с лучшим weighted avg f1 на валидации = {max_epoch_f1_val}')  
            print('Accuracy данной модели равно', acc_model) 

            mlflow.log_metric("max f1 saved model", max_epoch_f1_val)
            mlflow.log_metric("accuracy of model", acc_model)

            mlflow.log_artifact(name_save_model)
    else:

        print('БЕЗ MLFLOW')
        model = model_CNN()
        if start_weight != None:
            model = torch.jit.load(start_weight)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=scheduler_step_size,
                                                        gamma=0.5)
        model = model.to(device)

        itr_record = 0
        max_epoch_acc_val = 0
        EPOCHS = epochs

        for epoch in range(EPOCHS):
            # Обнуление градиентов параметров модели
            model.train()
            running_loss_train = 0
            running_corrects_train = 0
            for itr, (inputs, targets) in enumerate(dataloader_train):
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_temp_size = inputs.shape[0]

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss_train += loss / batch_temp_size
                running_corrects_train += accuracy(outputs, targets)


            epoch_loss_train = running_loss_train / len(dataloader_train)
            epoch_acc_train = running_corrects_train / len(dataloader_train)

            # Валидация модели после каждой эпохи
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                running_corrects_val = 0
                for i, (inputs, targets) in enumerate(dataloader_val):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    batch_temp_size = inputs.shape[0]

                    outputs = model(inputs)

                    val_loss += loss_func(outputs, targets).item() / batch_temp_size
                    running_corrects_val += accuracy(outputs, targets)  

                # Вычисление среднего значения функции потерь на валидационном наборе данных
                epoch_loss_val =  val_loss / len(dataloader_val)
                epoch_acc_val =  running_corrects_val / len(dataloader_val)

            if scheduler:
                lr_scheduler.step()
            # Вывод значения функции потерь на каждой эпохе

            print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss_train:.4f},'
                  f' Train Aсс: {epoch_acc_train:.4f}'
                  f' Val Loss: {epoch_loss_val:.4f}, Val Acc:{epoch_acc_val:.4f} ')
            
            if epoch >= 2 and epoch_acc_val > max_epoch_acc_val:
                max_epoch_acc_val = epoch_acc_val
                model_to_save = torch.nn.Sequential(
                    model,
                    torch.nn.Softmax(1),
                )

                model_scripted = torch.jit.script(model_to_save)
                name_save_model = directory_save + '/' + name_save +'.pt'
                model_scripted.save(name_save_model) 
                evaluate_model(model=model, data_loader=dataloader_val)
            
        print('Обучение завершено')
        print(f'Сохранена модель {name_save_model} с лучшим acc на валидации = {max_epoch_acc_val}')      
