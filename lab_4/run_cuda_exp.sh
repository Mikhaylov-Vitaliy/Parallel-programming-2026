#!/bin/bash
#SBATCH --job-name=cuda_auto_exp
#SBATCH --output=cluster_cuda_exp.out
#SBATCH --error=cluster_cuda_exp.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

cd $SLURM_SUBMIT_DIR
module purge

module load cuda

rm -f performance_report_cuda.txt result.txt

echo "=== ИНФОРМАЦИЯ ОБ УСТРОЙСТВЕ GPU ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo "====================================="

nvcc -O3 -std=c++11 matrix_cuda.cu -o ./cuda_matrix_exec

if [ ! -f ./cuda_matrix_exec ]; then
    echo "КРИТИЧЕСКАЯ ОШИБКА: Сборка matrix_cuda.cu не удалась!" >&2
    exit 1
fi

echo "=== СТАРТ АВТОМАТИЗИРОВАННЫХ ЭКСПЕРИМЕНТОВ CUDA ==="

SIZES=(200 400 800 1200 1600 2000)

BLOCK_SIZES=(4 8 16 32)

for b in "${BLOCK_SIZES[@]}"; do
    echo "----------------------------------------"
    echo "Запуск тестов с размером блока потоков: ${b}x${b} ($((b*b)) потоков/блок)"
    echo "----------------------------------------"
    
    for n in "${SIZES[@]}"; do
        echo ">> Расчет матрицы ${n}x${n} (блок ${b}x${b})..."
        ./cuda_matrix_exec $n $b
    done
done

echo "=== ВСЕ ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ ==="
