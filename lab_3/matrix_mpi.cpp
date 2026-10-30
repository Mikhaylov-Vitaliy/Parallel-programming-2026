#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>

typedef std::vector<double> Matrix1D;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;

    if (rank == 0) {
        if (argc < 2) {
            std::cerr << "Ошибка: Не указан размер матрицы N.\n";
            std::cerr << "Использование: mpirun -np <ядра> ./mpi_matrix_exec <размер_N>\n" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        n = std::atoi(argv[1]);
        if (n <= 0 || n > 2000) {
            std::cerr << "Ошибка: Некорректный размер матрицы (должен быть от 1 до 2000)." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    int sum = 0;
    for (int i = 0; i < size; ++i) {
        int rows = n / size;
        if (i < n % size) rows++; // Остаток строк отдаем первым процессам
        sendcounts[i] = rows * n;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    int local_num_rows = sendcounts[rank] / n;

    Matrix1D A;
    Matrix1D B(n * n);
    Matrix1D C;

    if (rank == 0) {
        A.resize(n * n);
        C.resize(n * n);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-10.0, 10.0);

        for (int i = 0; i < n * n; ++i) {
            A[i] = dis(gen);
            B[i] = dis(gen);
        }
    }

    // Локальные буферы для хранения порции данных текущего процесса
    Matrix1D local_A(sendcounts[rank]);
    Matrix1D local_C(sendcounts[rank], 0.0);

    // Синхронизация перед замером времени алгоритма
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Рассылаем порции строк матрицы А
    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_A.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 2. Рассылаем всю матрицу B всем процессам
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 3. Параллельное локальное умножение (с оптимизацией кеша i-k-j)
    for (int i = 0; i < local_num_rows; ++i) {
        for (int k = 0; k < n; ++k) {
            double val = local_A[i * n + k];
            for (int j = 0; j < n; ++j) {
                local_C[i * n + j] += val * B[k * n + j];
            }
        }
    }

    // 4. Сборка результатов в итоговую матрицу C на процессе 0
    MPI_Gatherv(local_C.data(), sendcounts[rank], MPI_DOUBLE,
        C.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    if (rank == 0) {
        std::ofstream outfile("result.txt");
        if (outfile.is_open()) {
            outfile << "Результат умножения матриц (Размер " << n << " x " << n << "):\n";
            outfile << "Количество MPI-процессов: " << size << "\n";
            outfile << std::fixed << std::setprecision(4);
            outfile << "Время расчета: " << duration.count() << " мс\n\n";

            outfile << std::setprecision(2);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    outfile << C[i * n + j] << "\t";
                }
                outfile << "\n";
            }
            outfile.close();
        }


        long long totalElements = static_cast<long long>(n) * n;
        long long ramBytes = totalElements * 3 * sizeof(double);
        double ramMegabytes = static_cast<double>(ramBytes) / (1024.0 * 1024.0);
        long long totalOps = 2 * static_cast<long long>(n) * n * n;
        double seconds = duration.count() / 1000.0;

        double gflops = (seconds > 0.0) ? ((static_cast<double>(totalOps) / seconds) / 1000000000.0) : 0.0;

        std::ofstream reportFile("performance_report.txt", std::ios_base::app);

        std::ostream* outputs[2];
        outputs[0] = &std::cout;
        outputs[1] = reportFile.is_open() ? &reportFile : &std::cout;

        for (int idx = 0; idx < 2; ++idx) {
            if (idx == 1 && !reportFile.is_open()) continue;

            std::ostream& out = *outputs[idx];

            out << "\n========================================\n";
            out << "ПАРАЛЛЕЛЬНЫЙ ОТЧЕТ ОБ ОБЪЁМЕ И ПРОИЗВОДИТЕЛЬНОСТИ (" << n << "x" << n << "):\n";
            out << "Использовано MPI-ядер: " << size << "\n";
            out << "========================================\n";
            out << "1. Математическая сложность и Скорость:\n";
            out << "   - Всего элементов в одной матрице: " << totalElements << "\n";
            out << "   - Всего операций (FLOP): ~" << totalOps << " оп.\n";
            out << std::fixed << std::setprecision(4);
            out << "   - Время расчета алгоритма: " << duration.count() << " мс (" << seconds << " сек)\n";
            out << std::setprecision(2);
            out << "   - ПРОИЗВОДИТЕЛЬНОСТЬ: " << gflops << " GFLOPS\n";
            out << "\n2. Затраты памяти (ОЗУ):\n";
            out << "   - Суммарная память под 3 матрицы: " << ramMegabytes << " МБ\n";
            out << "========================================\n";
        }

        if (reportFile.is_open()) {
            reportFile.close();
        }
    }

    MPI_Finalize();
    return 0;
}

