import subprocess
import re

EXECUTABLE = 'mandelbrot.exe'
THREADS_LIST = [1, 2, 3, 4, 5, 6, 7, 8]
VIEWS = [1, 2]

def run_benchmark(view_id):
    speedups = []
    print(f"=== Запуск замеров для View {view_id} ===")
    
    for t in THREADS_LIST:
        # Запускаем ./mandelbrot -t <t> -v <view_id>
        result = subprocess.run(
            [EXECUTABLE, '-t', str(t), '-v', str(view_id)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Ошибка при запуске с -t {t} -v {view_id}: {result.stderr}")
            speedups.append(0.0)
            continue
            
        # Ищем в выводе программы строку вида "(3.45x speedup from 4 threads)"
        match = re.search(r'\((\d+\.\d+)x speedup', result.stdout)
        if match:
            speedup = float(match.group(1))
            speedups.append(speedup)
            print(f"Потоков: {t} -> Ускорение: {speedup:.2f}x")
        else:
            print(f"Не удалось распознать вывод для t={t}:\n{result.stdout}")
            speedups.append(0.0)
            
    return speedups

if __name__ == '__main__':
    v1_results = run_benchmark(1)
    print()
    v2_results = run_benchmark(2)
    
    print("\n--- Готовые массивы для вставки в plot_speedup.py ---")
    print(f"view1_speedup = {v1_results}")
    print(f"view2_speedup = {v2_results}")
