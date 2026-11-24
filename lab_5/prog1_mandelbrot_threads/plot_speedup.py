import matplotlib.pyplot as plt

# Количество потоков
threads = [1, 2, 3, 4, 5, 6, 7, 8]

# Идеальное линейное ускорение (y = x)
ideal_speedup = threads

# --- ЗАМЕНИТЕ ЭТИ ДАННЫЕ НА ВАШИ ЗАМЕРЫ ИЗ КОНСОЛИ ---
# Ускорение для блочного метода (Этап 2)
block_view1 = [1.0, 1.92, 1.65, 2.38, 2.60, 3.15, 3.40, 3.95]
block_view2 = [1.0, 1.68, 2.15, 2.75, 3.10, 3.60, 4.10, 4.55]

# Ускорение для метода с чередованием строк (Этап 4)
interleaved_view1 = [1.0, 1.98, 2.94, 3.89, 4.80, 5.72, 6.61, 7.45]
interleaved_view2 = [1.0, 1.97, 2.91, 3.85, 4.76, 5.68, 6.55, 7.38]
# ---------------------------------------------------

plt.figure(figsize=(10, 6), dpi=150)

# Идеальное ускорение
plt.plot(threads, ideal_speedup, linestyle='--', color='gray', linewidth=1.5, label='Идеальное линейное ускорение')

# Блочное разделение
plt.plot(threads, block_view1, marker='o', linewidth=2, color='#d62728', label='Блочный метод (View 1)')
plt.plot(threads, block_view2, marker='s', linewidth=2, color='#ff7f0e', label='Блочный метод (View 2)')

# Чередование строк
plt.plot(threads, interleaved_view1, marker='^', linewidth=2, color='#1f77b4', label='Чередование строк (View 1)')
plt.plot(threads, interleaved_view2, marker='D', linewidth=2, color='#2ca02c', label='Чередование строк (View 2)')

# Оформление графика
plt.title('Ускорение генерации множества Мандельброта в зависимости от числа потоков', fontsize=12, pad=12)
plt.xlabel('Количество потоков (numThreads)', fontsize=11)
plt.ylabel('Ускорение (Speedup, раз)', fontsize=11)
plt.xticks(threads)
plt.yticks(range(1, 9))
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left', fontsize=10)

# Подпись характерного провала на 3 потоках для отчета
plt.annotate('Провал из-за дисбаланса\nнагрузки (центральный блок)', 
             xy=(3, block_view1[2]), 
             xytext=(3.4, 1.3),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
             fontsize=9, color='#d62728')

plt.tight_layout()
plt.savefig('mandelbrot_speedup.png')
plt.show()
