#!/bin/bash
#SBATCH --job-name=mpi_auto_exp
#SBATCH --output=cluster_experiment.out
#SBATCH --error=cluster_experiment.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:25:00

# Переходим в рабочую директорию
cd $SLURM_SUBMIT_DIR

# Полностью очищаем окружение
module purge

# Загружаем ТОЛЬКО модуль MPI (БЕЗ компилятора icc18, чтобы не ломать пути к библиотекам)
module load intel/mpi5

# Очищаем старый отчет
rm -f performance_report.txt

# Явно заставляем Intel MPI использовать стандартный системный компилятор GNU G++
export I_MPI_CXX=g++

# Компилируем через mpic++ (альтернативный макрос сборки MPI, работающий с g++)
mpic++ -O3 -std=c++11 matrix_mpi.cpp -o ./mpi_matrix_exec

# Проверка сборки
if [ ! -f ./mpi_matrix_exec ]; then
    echo "КРИТИЧЕСКАЯ ОШИБКА: Сборка не удалась через системный g++!" >&2
    exit 1
fi

echo "=== СТАРТ АВТОМАТИЗИРОВАННЫХ ЭКСПЕРИМЕНТОВ ==="

SIZES=(200 400 800 1200 1600 2000)
CORES=(1 2 4 8 12 16)

for p in "${CORES[@]}"; do
    echo "----------------------------------------"
    echo "Запуск тестов на количестве вычислительных ядер: $p"
    echo "----------------------------------------"
    
    for n in "${SIZES[@]}"; do
        echo ">> Расчет матрицы ${n}x${n} на ${p} ядрах..."
        srun -n $p ./mpi_matrix_exec $n
    done
done

echo "=== ВСЕ ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ ==="

