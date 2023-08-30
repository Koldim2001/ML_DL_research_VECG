import mne
import math
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import signal
from matplotlib.pyplot import figure
import scipy.signal
import neurokit2 as nk
import plotly.express as px
from shapely.geometry import Polygon
import plotly.graph_objects as go
import warnings
from sklearn.preprocessing import StandardScaler


def convert_to_posix_path(windows_path):
    # Перевод пути к формату posix:
    posix_path = windows_path.replace('\\', '/')
    return posix_path


def rename_columns(df):
    # Приводит к правильному виду данные в df:
    new_columns = []
    for column in df.columns:
        new_columns.append(column[:-4])
    df.columns = new_columns
    return df


def discrete_signal_resample(signal, time, new_sampling_rate):
    ## Производит ресемплирование
    # Текущая частота дискретизации
    current_sampling_rate = 1 / np.mean(np.diff(time))

    # Количество точек в новой дискретизации
    num_points_new = int(len(signal) * new_sampling_rate / current_sampling_rate)

    # Используем scipy.signal.resample для изменения дискретизации
    new_signal = scipy.signal.resample(signal, num_points_new)
    new_time = np.linspace(time[0], time[-1], num_points_new)

    return new_signal, new_time


def calculate_area(points):
    # Считает площадь замкнутого полигона
    polygon = Polygon(points)
    area_inside_loop = polygon.area
    return area_inside_loop


def find_mean(df_term):
    # Считает средние значения петель
    x_center = df_term.x.mean()
    y_center = df_term.y.mean()
    z_center = df_term.z.mean()
    return [x_center, y_center, z_center]


def find_qrst_angle(mean_qrs, mean_t, name=''):
    ## Находит угол QRST с помощью скалярного произведения
    # Преобразуем списки в numpy массивы
    mean_qrs = np.array(mean_qrs)
    mean_t = np.array(mean_t)

    # Находим угол между векторами в радианах
    dot_product = np.dot(mean_qrs, mean_t)
    norm_qrs = np.linalg.norm(mean_qrs)
    norm_t = np.linalg.norm(mean_t)
    angle_radians = np.arccos(dot_product / (norm_qrs * norm_t))

    # Конвертируем угол из радиан в градусы
    angle_degrees = np.degrees(angle_radians)
    #print(f"Угол QRST {name}равен {round(angle_degrees, 2)} градусов")

    return angle_degrees


def make_vecg(df_term):
    # Получает значения ВЭКГ из ЭКГ
    DI = df_term['ECG I']
    DII = df_term['ECG II']
    V1 = df_term['ECG V1']
    V2 = df_term['ECG V2']
    V3 = df_term['ECG V3']
    V4 = df_term['ECG V4']
    V5 = df_term['ECG V5']
    V6 = df_term['ECG V6']

    df_term['x'] = -(-0.172*V1-0.074*V2+0.122*V3+0.231*V4+0.239*V5+0.194*V6+0.156*DI-0.01*DII)
    df_term['y'] = (0.057*V1-0.019*V2-0.106*V3-0.022*V4+0.041*V5+0.048*V6-0.227*DI+0.887*DII)
    df_term['z'] = -(-0.229*V1-0.31*V2-0.246*V3-0.063*V4+0.055*V5+0.108*V6+0.022*DI+0.102*DII)
    return df_term

    
def loop(df_term, name, show=False):
    # Подсчет и отображение площади петли
    if name == 'T':
        name_loop = 'ST-T'
    else:
        name_loop = name

    if show:
        plt.figure(figsize=(15, 5), dpi=80)
        plt.subplot(1, 3, 1)
        plt.plot(df_term.x,df_term.y)
        plt.title('Фронтальная плоскость')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.subplot(1, 3, 2)
        plt.plot(df_term.y,df_term.z)
        plt.title('Сагиттальная плоскость')
        plt.xlabel('Y')
        plt.ylabel('Z')

        plt.subplot(1, 3, 3)
        plt.plot(df_term.x, df_term.z)
        plt.title('Аксиальная плоскость')  
        plt.xlabel('X')
        plt.ylabel('Z')

        plt.suptitle(f'{name_loop} петля', fontsize=16)
        plt.show()
    
    points = list(zip(df_term['x'], df_term['y']))
    area_inside_loop_1 = calculate_area(points)
    #print(f"Площадь петли {name_loop} во фронтальной плоскости:", area_inside_loop_1)

    points = list(zip(df_term['y'], df_term['z']))
    area_inside_loop_2 = calculate_area(points)
    #print(f"Площадь петли {name_loop} в сагиттальной плоскости:", area_inside_loop_2)

    points = list(zip(df_term['x'], df_term['z']))
    area_inside_loop_3 = calculate_area(points)
    #print(f"Площадь петли {name_loop} в аксиальной плоскости:", area_inside_loop_3)

    return area_inside_loop_1, area_inside_loop_2, area_inside_loop_3


def get_area(show, df, waves_peak, start, Fs_new, QRS, T):
    # Выделяет области петель для дальнейшей обработки - подсчета угла QRST и площадей
    area = []
    # Уберем nan:
    waves_peak['ECG_Q_Peaks'] = [x for x in waves_peak['ECG_Q_Peaks'] if not math.isnan(x)]
    waves_peak['ECG_S_Peaks'] = [x for x in waves_peak['ECG_S_Peaks'] if not math.isnan(x)]
    waves_peak['ECG_T_Offsets'] = [x for x in waves_peak['ECG_T_Offsets'] if not math.isnan(x)]   

    # QRS петля
    # Ищем ближний пик к R пику
    closest_Q_peak = min(waves_peak['ECG_Q_Peaks'], key=lambda x: abs(x - start))
    closest_S_peak = min(waves_peak['ECG_S_Peaks'], key=lambda x: abs(x - start))
    df_new = df.copy()
    df_term = df_new.iloc[closest_Q_peak:closest_S_peak,:]
    df_row = df_new.iloc[closest_Q_peak:closest_Q_peak+1,:]
    df_term = pd.concat([df_term, df_row])
    df_term = make_vecg(df_term)
    mean_qrs = find_mean(df_term)
    if QRS:
        area = list(loop(df_term, name='QRS', show=show))

    ## ST-T петля
    # Ищем ближний пик к R пику
    closest_S_peak = min(waves_peak['ECG_S_Peaks'], key=lambda x: abs(x - start))
    # Ищем ближний пик к S пику
    closest_T_end = min(waves_peak['ECG_T_Offsets'], key=lambda x: abs(x - closest_S_peak))
    df_new = df.copy()
    df_term = df_new.iloc[closest_S_peak + int(0.025*Fs_new) : closest_T_end, :]
    df_row = df_new.iloc[closest_S_peak+int(0.025*Fs_new):closest_S_peak+int(0.025*Fs_new)+1,:]
    df_term = pd.concat([df_term, df_row])
    df_term = make_vecg(df_term)
    mean_t = find_mean(df_term)
    if T:
        area.extend(list(loop(df_term, name='T', show=show)))
    return area, mean_qrs, mean_t


def preprocessing_3d(list_coord):
    # Строит линии на 3D графике, отвечающие за вектора средних ЭДС петель
    A = np.array(list_coord)

    step = 0.025
    # Создаем массив точек от (0, 0, 0) до точки A с заданным шагом
    interpolated_points = []
    for t in np.arange(0, 1, step):
        interpolated_point = t * A
        interpolated_points.append(interpolated_point)

    # Добавляем точку A в конец массива
    interpolated_points.append(A)

    # Преобразуем список точек в numpy массив
    interpolated_points = np.array(interpolated_points)

    df = pd.DataFrame(interpolated_points, columns=['x', 'y', 'z'])
    df['s']=20 # задали размер для 3D отображения
    return df


def angle_3d_plot(df1, df2, df3):
    # Построение интерактивного графика логов вычисления угла QRST 
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=df1['x'],
            y=df1['y'],
            z=df1['z'],
            mode='markers',
            marker=dict(size=df1['s'], sizemode='diameter', opacity=1),
            name='Средняя электродвижущая сила QRS'
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df2['x'],
            y=df2['y'],
            z=df2['z'],
            mode='markers',
            marker=dict(size=df2['s'], sizemode='diameter', opacity=1),
            name='Средняя электродвижущая сила ST-T'
        )
    )
    df3['size'] = 10
    fig.add_trace(
        go.Scatter3d(
            x=df3['x'],
            y=df3['y'],
            z=df3['z'],
            mode='markers',
            marker=dict(size=df3['size'], sizemode='diameter', opacity=1),
            name='ВЭКГ'
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()





#------------------------------------------ГЛАВНЫЙ КОД--------------------------------------#

def processing(data):
    # ------------------ ARG parse ------------------
    data_edf = data["data_edf"]
    n_term_start = data["n_term_start"]
    n_term_finish = data["n_term_finish"] 
    filt = data["filt"]
    f_sreza = data["f_sreza"]
    Fs_new = data["f_sampling"]
    show_detect_pqrst = data["show_detected_pqrst"]
    show_ECG = data["show_ecg"]
    plot_3D = data["plot_3d"]
    save_images = data["save_images"]
    show_log_scaling = data["show_log_scaling"]
    cancel_showing = data["cancel_showing"]
    QRS_loop_area = data["qrs_loop_area"]
    T_loop_area = data["t_loop_area"]
    show_log_loop_area = data["show_log_loop_area"]
    count_qrst_angle = data["count_qrst_angle"]
    show_log_qrst_angle = data["show_log_qrst_angle"]
    save_coord = data["save_coord"] 

    ## СЛЕДУЕТ УБРАТЬ ПРИ ТЕСТИРОВАНИИ:
    # Устанавливаем фильтр для игнорирования всех RuntimeWarning
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Включаем режим, позволяющий открывать графики сразу все
    plt.ion()

    if cancel_showing:
        show_detect_pqrst = False
        show_ECG = False
        plot_3D = False
        show_log_scaling = False
        show_log_loop_area = False
        show_log_qrst_angle = False

    if n_term_finish != None:
        if n_term_finish < n_term_start:
            raise ValueError("Ошибка: n_term_finish должно быть >= n_term_start")
        else:
          n_term = [n_term_start, n_term_finish]  
    else:
        n_term = n_term_start

    if '\\' in data_edf:
        # Преобразуем путь в формат Posix
        data_edf = convert_to_posix_path(data_edf)

    # Считывание edf данных:
    data = mne.io.read_raw_edf(data_edf, verbose=0)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    fd = info['sfreq'] # Частота дискретизации
    df = pd.DataFrame(data=raw_data.T,    # values
                index=range(raw_data.shape[1]),  # 1st column as index
                columns=channels)  # 1st row as the column names
    # Переименование столбцов при необходимости:
    if 'ECG I-Ref' in df.columns:
        df = rename_columns(df)
        channels = df.columns

    # Создание массива времени    
    Ts = 1/fd
    t = []
    for i in range(raw_data.shape[1]):
        t.append(i*Ts)

    # Ресемлинг:
    df_new = pd.DataFrame()
    for graph in channels:
        sig = np.array(df[graph])
        new_ecg, time_new = discrete_signal_resample(sig, t, Fs_new)
        df_new[graph] = pd.Series(new_ecg) 
    df = df_new.copy()

    # ФВЧ фильтрация артефактов дыхания:
    if filt == True:
        df_new = pd.DataFrame()
        for graph in channels:
            sig = np.array(df[graph])
            sos = scipy.signal.butter(1, f_sreza, 'hp', fs=Fs_new, output='sos')
            avg = np.mean(sig)
            filtered = scipy.signal.sosfilt(sos, sig)
            filtered += avg
            df_new[graph] = pd.Series(filtered)
        df = df_new.copy()
        
    # ФНЧ фильтрация (по желанию можно включить):
    filt_low_pass = False
    if filt_low_pass:
        df_new = pd.DataFrame()
        for graph in channels:
            sig = np.array(df[graph])
            sos = scipy.signal.butter(1, 100, 'lp', fs=Fs_new, output='sos')
            avg = np.mean(sig)
            filtered = scipy.signal.sosfilt(sos, sig)
            filtered += avg
            df_new[graph] = pd.Series(filtered)
        df = df_new.copy()

    ## Поиск точек PQRST:
    n_otvedenie = 'I'
    signal = np.array(df['ECG I'])  

    # способ чистить сигнал перед поиском пиков:
    signal = nk.ecg_clean(signal, sampling_rate=Fs_new, method="neurokit") 

    # Поиск R зубцов:
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=Fs_new)

    # Проверка в случае отсутствия результатов и повторная попытка:
    if rpeaks['ECG_R_Peaks'].size <= 5:
        print("На I отведении не удалось детектировать R зубцы")
        print("Проводим детектирование по II отведению:")
        n_otvedenie = 'II'
        signal = np.array(df['ECG II'])  
        signal = nk.ecg_clean(signal, sampling_rate=Fs_new, method="neurokit") 
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=Fs_new)
        
        # При повторной проблеме выход из функции:
        if rpeaks['ECG_R_Peaks'].size <= 3:
            print('Сигналы ЭКГ слишком шумные для анализа')
            # Отобразим эти шумные сигналы:
            if not cancel_showing:
                num_channels = len(channels)
                fig, axs = plt.subplots(int(num_channels/2), 2, figsize=(11, 8), sharex=True)
                for i, graph in enumerate(channels):
                    row = i // 2
                    col = i % 2
                    sig = np.array(df[graph])
                    axs[row, col].plot(time_new, sig)
                    axs[row, col].set_title(graph)
                    axs[row, col].set_xlim([0, 6])
                    axs[row, col].set_title(graph)
                    axs[row, col].set_xlabel('Time (seconds)')
                plt.tight_layout()
                plt.show()
                plt.ioff()
                plt.show()
            return # Выход из функции досрочно

    # Поиск точек pqst:
    _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=Fs_new, method="peak")

    # Отображение PQST точек на сигнале первого отведения (или второго при ошибке на первом)
    if show_detect_pqrst:
        plt.figure(figsize=(12, 5))

        # Отобразим сигнал на графике
        plt.plot(time_new, signal, label='Signal', color='black')

        # Отобразим вертикальные линии для каждого типа точек
        for wave_type, peaks in waves_peak.items():
            if wave_type in ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']:
                wave_type_label = wave_type.split('_')[1]  # Извлекаем часть имени для метки графика
                for peak in peaks:
                    if not np.isnan(peak):  # Проверяем, что значение точки не является NaN
                        if wave_type == 'ECG_P_Peaks':
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted',
                                        color='red', label=f'{wave_type_label} Peak')
                        elif wave_type == 'ECG_Q_Peaks':
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted',
                                        color='green', label=f'{wave_type_label} Peak')
                        elif wave_type == 'ECG_S_Peaks': 
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted',
                                        color='m', label=f'{wave_type_label} Peak')
                        else:  
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted',
                                        color='blue', label=f'{wave_type_label} Peak')
        plt.xlim([0.5, 6])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Signal ECG I')
        plt.title(f'Детекция PQRST на {n_otvedenie} отведении')
        plt.show()

    # Отображение многоканального ЭКГ с детекцией R зубцов
    if show_ECG:
        num_channels = len(channels)
        fig, axs = plt.subplots(int(num_channels/2), 2, figsize=(11, 8), sharex=True)

        for i, graph in enumerate(channels):
            row = i // 2
            col = i % 2

            sig = np.array(df[graph])

            axs[row, col].plot(time_new, sig)
            axs[row, col].scatter(time_new[rpeaks['ECG_R_Peaks']], 
                                  sig[rpeaks['ECG_R_Peaks']], color='red')
            axs[row, col].set_title(graph)
            axs[row, col].set_xlim([0, 6])
            axs[row, col].set_title(graph)
            axs[row, col].set_xlabel('Time (seconds)')

        plt.tight_layout()
        plt.show()

    # Выбор исследуемого периода/периодов
    i = n_term
    if type(i) == list:
        print(f"Запрошен диапазон с {i[0]} по {i[1]} период включительно")
        fin = i[1]
        beg = i[0]
    else:
        #print(f"Запрошен {i} период")
        fin = i
        beg = i

    if beg-1 < 0 or fin >= len(rpeaks['ECG_R_Peaks']):
        #print('Запрашиваемого перода/диапазона периодов не существует')
        return # Выход из функции досрочно
    
    start = rpeaks['ECG_R_Peaks'][beg-1]
    end = rpeaks['ECG_R_Peaks'][fin]
    df_term = df.iloc[start:end,:]
    df_row = df.iloc[start:start+1,:]
    df_term = pd.concat([df_term, df_row])

    # Расчет ВЭКГ
    df_term = make_vecg(df_term)
    df_term['size'] = 100 # задание размера для 3D визуализации

    # Построение проекций ВЭКГ:
    if not cancel_showing:
        plt.figure(figsize=(15, 5), dpi=90)
        plt.subplot(1, 3, 1)
        plt.plot(df_term.x,df_term.y)
        plt.title('Фронтальная плоскость')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.subplot(1, 3, 2)
        plt.plot(df_term.y,df_term.z)
        plt.title('Сагиттальная плоскость')
        plt.xlabel('Y')
        plt.ylabel('Z')

        plt.subplot(1, 3, 3)
        plt.plot(df_term.x, df_term.z)
        plt.title('Аксиальная плоскость')  
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.show()

    # Интерактивное 3D отображение
    if plot_3D:
        fig = px.scatter_3d(df_term, x='x', y='y', z='z', size='size', size_max=10, opacity=1)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

    # Работа при указании одного периода ЭКГ: 
    if  n_term_finish == None or n_term_finish == n_term_start:

        if save_coord:
            df_save = df_term[['x', 'y', 'z']]
            # Путь к файлу CSV для сохранения
            file_name_without_extension = os.path.splitext(os.path.basename(data_edf))[0]
            name = f'{file_name_without_extension}_period_{n_term_start}.csv'

            # Сохраняем выбранные столбцы в CSV файл
                # Создадим папки для записи если их еще нет:
            if not os.path.exists('point_cloud_dataset'):
                os.makedirs('point_cloud_dataset')
            df_save.to_csv('point_cloud_dataset/' + name, index=False)

            #### Еще нормализованные данные сохраним:
            df_selected = df_term[['x', 'y', 'z']]
            # Нормализуем данные
            scaler = StandardScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df_selected), columns=['x', 'y', 'z'])

            # Путь к файлу CSV для сохранения
            file_name_without_extension = os.path.splitext(os.path.basename(data_edf))[0]
            name = f'{file_name_without_extension}_period_{n_term_start}_normalized.csv'

            # Создаем папку для записи, если её еще нет
            output_folder = 'point_cloud_dataset_normalized'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Сохраняем нормализованные данные в CSV файл
            output_path = os.path.join(output_folder, name)
            df_normalized.to_csv(output_path, index=False)

            return df_save.shape[0]


        ## Масштабирование:
        # Поиск центра масс:
        x_center = df_term.x.mean()
        y_center = df_term.y.mean()
        z_center = df_term.z.mean()

        df_term['x_scaled'] = df_term.x - x_center
        df_term['y_scaled'] = df_term.y - y_center
        df_term['z_scaled'] = df_term.z - z_center

        # Нормирование на максимальное значение 
        max_value = max(df_term['x_scaled'].abs().max(),
                        df_term['y_scaled'].abs().max(),
                        df_term['z_scaled'].abs().max())
        df_term['x_scaled'] = df_term['x_scaled'] / max_value
        df_term['y_scaled'] = df_term['y_scaled'] / max_value
        df_term['z_scaled'] = df_term['z_scaled'] / max_value

        # Показ логов масштабирования
        if show_log_scaling:
            plt.figure(figsize=(8, 10), dpi=80)
            plt.subplot(3, 2, 1)
            plt.plot(df_term.x, df_term.y)
            plt.title('Исходные проекции')
            plt.xlabel('X')
            plt.ylabel('Y') 
            plt.plot(x_center, y_center, marker='*', markersize=11, label='Центр масс', color='red')
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 2)
            plt.plot(df_term.x_scaled, df_term.y_scaled)
            plt.title('Масштабированные проекции')
            plt.xlabel('X')
            plt.ylabel('Y') 
            plt.xlim([-1.05, 1.05])
            plt.ylim([-1.05, 1.05])
            plt.grid(True)

            plt.subplot(3, 2, 3)
            plt.plot(df_term.y, df_term.z)
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.plot(y_center, z_center, marker='*', markersize=11, label='Центр масс', color='red')
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 4)
            plt.plot(df_term.y_scaled, df_term.z_scaled)
            plt.xlim([-1.05, 1.05])
            plt.ylim([-1.05, 1.05])
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.grid(True)

            plt.subplot(3, 2, 5)
            plt.plot(df_term.x, df_term.z)
            plt.xlabel('X')
            plt.ylabel('Z')
            plt.plot(x_center, z_center, marker='*', markersize=12, label='Центр масс', color='red')
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 6)
            plt.plot(df_term.x_scaled, df_term.z_scaled)
            plt.xlabel('X')
            plt.ylabel('Z')
            plt.xlim([-1.05, 1.05])
            plt.ylim([-1.05, 1.05])
            plt.grid(True)
            plt.show()

        # Поиск площадей при задании на исследование одного периодка ЭКГ:
        area_projections = []
        angle_qrst = []
        angle_qrst_front = []
        if QRS_loop_area or T_loop_area:
            area_projections , mean_qrs, mean_t = get_area(show=show_log_loop_area, df=df,
                                                       waves_peak=waves_peak, start=start,
                                                       Fs_new=Fs_new,  QRS=QRS_loop_area, 
                                                       T=T_loop_area)
        # Определение угла QRST:
        if count_qrst_angle:
            angle_qrst = find_qrst_angle(mean_qrs, mean_t)
            angle_qrst_front = find_qrst_angle(mean_qrs[:2], mean_t[:2],
                                               name='во фронтальной плоскости ')

            # Отображение трехмерного угла QRST
            if show_log_qrst_angle:
                df_qrs = preprocessing_3d(mean_qrs)
                df_t = preprocessing_3d(mean_t)
                angle_3d_plot(df_qrs, df_t, df_term)
    

    # Сохранение масштабированных изображений
    if save_images and (n_term_finish == None or n_term_finish == n_term_start):
        file_name_without_extension = os.path.splitext(os.path.basename(data_edf))[0]
        name = f'{file_name_without_extension}_period_{n_term_start}.png'
        
        # Создадим папки для записи если их еще нет:
        if not os.path.exists('saved_vECG'):
            os.makedirs('saved_vECG')
        if not os.path.exists('saved_vECG/frontal_plane'):
            os.makedirs('saved_vECG/frontal_plane')
        if not os.path.exists('saved_vECG/sagittal_plane'):
            os.makedirs('saved_vECG/sagittal_plane')
        if not os.path.exists('saved_vECG/axial_plane'):
            os.makedirs('saved_vECG/axial_plane')      

        # После каждого plt.show() добавим код для сохранения графика в ЧБ формате
        plt.figure(figsize=(7, 7), dpi=150)
        plt.xlim([-1.03, 1.03])
        plt.ylim([-1.03, 1.03])
        plt.plot(df_term.x_scaled, df_term.y_scaled, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/frontal_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()

        plt.figure(figsize=(7, 7), dpi=150)
        plt.xlim([-1.03, 1.03])
        plt.ylim([-1.03, 1.03])
        plt.plot(df_term.y_scaled, df_term.z_scaled, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/sagittal_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()

        plt.figure(figsize=(7, 7), dpi=150)
        plt.xlim([-1.03, 1.03])
        plt.ylim([-1.03, 1.03])  
        plt.plot(df_term.x_scaled, df_term.z_scaled, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/axial_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()
        #print('Фотографии сохранены в папке saved_vECG')

    # Выключаем интерактивный режим, чтобы окна графиков не закрывались сразу
    plt.ioff()
    plt.show()
    
    if  n_term_finish == None or n_term_finish == n_term_start:
        return   area_projections, angle_qrst, angle_qrst_front




















##########################################################################

import os
import pandas as pd
import random



def random_split(path, x, eps):
    print('\nВыбран режим автоматического сплитования:')
    # Загрузка CSV-файла в DataFrame
    dtype_mapping = {'Исходное название изображения': str}  # Задаем тип "str" для первого столбца
    df = pd.read_csv(path, dtype=dtype_mapping)

    # Вычисление текущей суммы
    current_sum = 0
    iter = 0
    # Проходим по строкам и изменяем значения, пока сумма не приблизится к x
    while current_sum < x*(1-0.01*eps):
        iter += 1
        if iter > 50000:
            print('!!!!\nНеудачная попытка разделить на train/val автоматически',
            '\nНе вышло сделать так, чтобы значение суммарного числа кропов на val удовлетворило',
            f' условию +- {eps}%  от требуемого {x} - [{round(x*(1-0.01*eps))},{round(x*(1+0.01*eps))}]')

            val_crops = df[df["Split train/val"] == "val"]["Общее число кропов"].sum()
            print('\nВ результате автоматического сплитования получилось отнести на валидацию',
                  f'{val_crops} кропов')
            percent_val = val_crops/df["Общее число кропов"].sum() * 100
            print(f'Итоговое соотношение train/val = {round(100-percent_val)}/{round(percent_val)}')
            print('Попробуй еще раз запустить get_csv.py!')
            df.to_csv(path, index=False)
            return

        random_row_index = random.randint(0, len(df) - 1)
        row = df.iloc[random_row_index]
        
        if row['Split train/val'] == 'train':
            delta = row['Общее число кропов'] 
            
            if current_sum + delta < x*(1+0.01*eps):
                df.loc[random_row_index, 'Split train/val'] = 'val'
                current_sum += delta

    # Выводим сумму кропов в валидационной выборке
    val_crops = df[df["Split train/val"] == "val"]["Общее число кропов"].sum()
    print(f'Условие рандомной генерации +-{eps}% от',
          f'требуемого {x} числа кропов на валидацию -',
          f'[{round(x*(1-0.01*eps))},{round(x*(1+0.01*eps))}]')
    print('\nВ результате автоматического сплитования получилось отнести на',
          f'валидацию {val_crops} кропов')
    percent_val = val_crops/df["Общее число кропов"].sum() * 100
    print(f'Итоговое соотношение train/val = {round(100-percent_val)}/{round(percent_val)}')

    # Сохраняем результат в новый CSV-файл
    df.to_csv(path, index=False)


def get_csv(data):
    # ------------------ ARG parse ------------------
    folder_dataset = data["folder_dataset"]
    folder_save = data["folder_save_csv"]
    auto_split = data["auto_split"]
    eps = data["percent_error"]
    percent_train = data["percent_train"]

    summa_crops = {}

    if not os.path.exists(folder_save):
        os.makedirs(folder_save)

    folder_path = folder_dataset
    photo_counts = {}
    extra_names = {}
    # Обход всех файлов в папке
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):  # Проверяем, что файлы являются PNG-изображениями (можно изменить формат по необходимости)
            base_name = file_name.split('_period')[0]  # Извлекаем исходное название файла
            name_crop = 'period' + file_name.split('.png')[0].split('period')[1]
            if base_name not in photo_counts:
                photo_counts[base_name] = 0
                extra_names[base_name] = {}
                extra_names[base_name][name_crop] = 0
            photo_counts[base_name] += 1
            if name_crop not in extra_names[base_name]:
                extra_names[base_name][name_crop] = 0
            extra_names[base_name][name_crop] += 1
    
        if file_name.endswith(".csv"):  # Проверяем, что файлы являются csv
            base_name = file_name.split('_period')[0]  # Извлекаем исходное название файла
            name_crop = 'period' + file_name.split('.csv')[0].split('period')[1]
            if base_name not in photo_counts:
                photo_counts[base_name] = 0
                extra_names[base_name] = {}
                extra_names[base_name][name_crop] = 0
            photo_counts[base_name] += 1
            if name_crop not in extra_names[base_name]:
                extra_names[base_name][name_crop] = 0
            extra_names[base_name][name_crop] += 1

    # Создаем DataFrame из словаря
    df = pd.DataFrame.from_dict(extra_names, orient='index')

    # Заполняем пропущенные значения нулями
    df = df.fillna(int(0))

    # Сортируем столбцы по алфавиту
    df = df.reindex(sorted(df.columns), axis=1)

    # Преобразуем столбцы с цифрами в тип int
    df = df.astype(int)

    # Добавляем столбец с суммой по всем столбцам кропов
    df['Общее число кропов'] = df.sum(axis=1)

    # Добавляем столбец "split" со значением "train"
    df['Split train/val'] = 'train'

    # Устанавливаем индексацию
    df = df.reset_index().rename(columns={'index': 'Исходное название изображения'})

    # Храним суммарное число кропов
    summa_crops = df["Общее число кропов"].sum()

    # Сохраняем DataFrame в CSV-файл
    name_save = folder_save + '/' + 'info_dataset.csv'
    df.to_csv(name_save, index=False)

    print(f'CSV таблица сохранена в папке {folder_save}')

    print('')
    if percent_train == None:
        percent_train = float(input('Введите процент % данных, которые '
                                'хотите положить на train - '))

    print(f'Всего имеется {summa_crops} кропов.')
    file_name = name_save
    print(f'Для реализации пропорции {int(percent_train)}/{int(100-percent_train)}'
                f' необходимо в файле {file_name} отнести на валидацию (val) '
                f'примерно {int(float(summa_crops)*(1-0.01*percent_train))} кропов.')
    
    if auto_split:
        random_split(name_save, x=int(float(summa_crops)*(1-0.01*percent_train)), eps=eps)
    else:
        print('\nТак как автоматический режим не включен, поэтому распределить на валидацию',
              'кропы надо в ручном режиме')


















######################################################################
import pandas as pd
import os 
import shutil


def get_train_val_images(csv_file):
    dtype_mapping = {'Исходное название изображения': str}  # Задаем тип "str" для первого столбца
    df = pd.read_csv(csv_file, dtype=dtype_mapping)

    train_images = df.loc[df['Split train/val'] == 'train', 'Исходное название изображения'].tolist()
    val_images = df.loc[df['Split train/val'] == 'val', 'Исходное название изображения'].tolist()
    return train_images, val_images



def split(data):
    # ------------------ ARG parse ------------------
    dataset_path = data["dataset_path"]
    path_final = data["splitted_dataset_name"]
    folder_csv = data["csv_folder"]
    excel_path = data["excel_file"]
    
    path_final_val = path_final +'/val'
    path_final_train = path_final +'/train'

    if not os.path.exists(path_final):
        os.makedirs(path_final)
    else: 
        if os.path.exists(path_final_val):
            shutil.rmtree(path_final_val)
        if os.path.exists(path_final_train):
            shutil.rmtree(path_final_train)

    os.makedirs(path_final_train)
    os.makedirs(path_final_val)

    for csv_file in os.listdir(folder_csv):
        csv_path = folder_csv + '/' + csv_file
        
        train_images, val_images = get_train_val_images(csv_path)

        if train_images==[]:
            raise Exception(f'Не обнаружено ни одного объекта train в {csv_path}')

        if val_images==[]:
            raise Exception(f'Не обнаружено ни одного объекта val в {csv_path}')

        if not os.path.exists(dataset_path):
            raise Exception(f'Датасет {dataset_path} с изображениями не найден')
        
        if not os.path.exists(excel_path):
            raise Exception(f'Иксель файл {excel_path} не найден') 

        df = pd.read_excel(excel_path)
 

        # Создаем список для хранения результатов в виде словарей
        result_data_train = []   
        result_data_val = []  
        list_not_found_val = [] 
        list_not_found_train = [] 

        for image in os.listdir(dataset_path):
            file_name = image.split('_period')[0]
            if file_name in train_images:
                if int(file_name) in df['FileID'].values:
                    # Получаем значение "KCl" из столбца df
                    EF_value = df[df['FileID'] == int(file_name)]['EF'].values[0]
                    
                    # Создаем словарь для текущего файла
                    new_row = {'File_Name': image, 'EF': EF_value}
                    
                    # Добавляем словарь в список результатов
                    result_data_train.append(new_row)

                    # Копируем файл в нужную папку
                    shutil.copy2(dataset_path + '/' + image, path_final_train)
                else:
                    list_not_found_train.append(file_name)
            if file_name in val_images:
                if int(file_name) in df['FileID'].values:
                    # Получаем значение "KCl" из столбца df
                    EF_value = df[df['FileID'] == int(file_name)]['EF'].values[0]
                    
                    # Создаем словарь для текущего файла
                    new_row = {'File_Name': image, 'EF': EF_value}
                    
                    # Добавляем словарь в список результатов
                    result_data_val.append(new_row)

                    # Копируем файл в нужную папку
                    shutil.copy2(dataset_path + '/' + image, path_final_val)
                else:
                    list_not_found_val.append(file_name)
                    #print(f'Файл {file_name} в иксель таблице не найден')
    
    # Создаем DataFrame из списка словарей
    result_df_train = pd.DataFrame(result_data_train)
    result_df_val = pd.DataFrame(result_data_val)

    result_df_train.to_csv(path_final_train + '/ground_truth.csv', index=False)
    result_df_val.to_csv(path_final_val + '/ground_truth.csv', index=False)

    if list_not_found_train:
        print('WARNING: Были однаружены изображения в train датасете, которые не имеют',
              f'информации о процентном соотношении в иксель файле - {list(set(list_not_found_train))}\n')
    
    if list_not_found_val:
        print('WARNING: Были однаружены изображения в val датасете, которые не имеют',
              f'информации о процентном соотношении в иксель файле - {list(set(list_not_found_train))}\n')

    val_amount = len(os.listdir(path_final + '/val')) - 1
    train_amount = len(os.listdir(path_final + '/train')) - 1
    print(f'Cоздано следующее число изображений:')
    print(f'На train - {int(train_amount)}')
    print(f'На val - {int(val_amount)} \n')

    percent_val = val_amount/(train_amount + val_amount) * 100
    print(f'Итоговое соотношение train/val = {round(100-percent_val)}/{round(percent_val)}\n')

    print(f'Разделенный датасет расположен в папке {path_final}')