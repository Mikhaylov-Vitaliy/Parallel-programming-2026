#!/bin/bash
#SBATCH --job-name=mpi_auto_exp
#SBATCH --output=cluster_experiment.out
#SBATCH --error=cluster_experiment.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:25:00

cd $SLURM_SUBMIT_DIR
module purge

module load intel/mpi5

rm -f performance_report.txt

# Явно заставляем Intel MPI использовать стандартный системный компилятор GNU G++
export I_MPI_CXX=g++

mpic++ -O3 -std=c++11 matrix_mpi.cpp -o ./mpi_matrix_exec

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

