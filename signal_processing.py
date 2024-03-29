import matplotlib.pyplot as plt
import numpy as np


def sliding_window_filter(sign):  # Реализация фильтрации сигнала методом скользящего окна
    window_size = 5  # длина окна
    filt_sign = []  # список, куда будут записанны значения отфильтрованного сигнала
    for i in range(len(sign) - window_size + 1):  # Берём фрагмент из первых winodw_size значений списка в качетсве окна
        window = sign[i:i + window_size]  # Рассчитываем среднее фрагмента, добавляем в результирующий список
        window_average = sum(window) / window_size  # Перебираем весь сигнал, сдвигая окно на 1 по исходному списку
        filt_sign.append(window_average)
    return filt_sign  # Возвращаем отфильтрованный сигнал


def add_zeros(array):  # дополнение исходного массива 0 для БПФ
    next_power_of_2 = int(2 ** np.ceil(np.log2(len(array))))  # вычисляем ближайшее к длине входного списка число сверху
    # которое будет являться двойкой в какой то степени
    padded_array = np.zeros(next_power_of_2)  # создаём новый спискок, заполняем 0
    # в соответсвии с вычисленным числом next_power_of_2
    padded_array[:len(array)] = array  # перезаписываем в новый массив значения из старого
    return padded_array  # получаем дополненный нулями исходный массив


def ACF_FFT(array):  # Расчёт АКФ с помощью БПФ
    ar_fft = np.fft.fft(array)  # искользуем БПФ
    ar_fft_conj = np.conj(ar_fft)  # вычисляем косплексно сопряжённое
    SPM = list(ar_fft * ar_fft_conj)  # перемножаем, получая СПМ
    acf = np.fft.ifft(SPM)  # возвращаем во временную область с помощью ОДПФ, получая АКФ
    return np.real(acf)


with open('test.txt', 'r', encoding='utf-8') as file:  # открываем тестовый файл (сигнал ЭКГ с помехой 50 Гц)
    signal = np.array([float(i) for i in file.readlines()])  # считываем дискретные отсчёты
    X_signal = np.array(range(1, len(signal) + 1))  # массив аргументов для дискретных отчётов исходного сигнала
filt_signal = sliding_window_filter(signal)  # фильтруем сигнал
X_filt_signal = np.array(range(1, len(filt_signal) + 1))  # массив аргументов для дискретных отчётов фильтрованного
# отчётов фильтрованного сигнала

plt.subplot(2, 2, 1)  # Задания поля графиков исходного и фильтрованного сигналов
plt.plot(X_signal, signal, label='Исходный сигнал')
plt.plot(X_filt_signal, filt_signal, color='orange', label='Отфильтрованный сигнал')  # График исходного сигнала
plt.title('Сигнал')
plt.legend()
plt.grid()


signal_var = np.square(signal - np.mean(signal))  # Расчёт временного ряда дисперсии
filt_sign_var = np.square(filt_signal - np.mean(signal))

plt.subplot(2, 2, 2)  # Задания поля графиков дисперсий двух версий сигнала
plt.plot(X_filt_signal, filt_sign_var)
plt.title('Распределение отклонений')
plt.grid()
plt.tight_layout()

filt_signal_zeros = add_zeros(filt_signal)  # Аналогично для отфильтрованного сигнала
TCF_filt = np.array(range(2 * len(filt_signal_zeros) - 1))
X_filt_signal_zeros = list(range(len(filt_signal_zeros)))
ACF_standart_filt = np.correlate(filt_signal_zeros, filt_signal_zeros, mode='full')
ACF_fft_filt = ACF_FFT(filt_signal_zeros)

plt.subplot(2, 2, 3)  # Задания поля графиков АКФ исходного сигнала
plt.plot(TCF_filt, ACF_standart_filt, label='АКФ отфильтрованого сигнала')  # Построение графика АКФ
plt.title('АКФ сигнала')
plt.grid()
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.plot(X_filt_signal_zeros, ACF_fft_filt, label='АКФ отфильтрованого сигнала')
plt.title('АКФ сигнала, реализованная при помощи БПФ')
plt.grid()
plt.tight_layout()

plt.show()
