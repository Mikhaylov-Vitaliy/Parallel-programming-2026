#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Ошибка CUDA в " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


__global__ void matrixMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Ошибка: Не указан размер матрицы N.\n";
        std::cerr << "Использование: ./cuda_matrix_exec <размер_N> [размер_блока_TILE]\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    if (n <= 0 || n > 10000) {
        std::cerr << "Ошибка: Некорректный размер матрицы (должен быть от 1 до 10000)." << std::endl;
        return 1;
    }

    int blockSize = (argc >= 3) ? std::atoi(argv[2]) : 16;
    if (blockSize <= 0 || blockSize > 32) {
        std::cerr << "Ошибка: Размер стороны блока должен быть от 1 до 32 (макс. 1024 потока на блок)." << std::endl;
        return 1;
    }

    size_t bytes = static_cast<size_t>(n) * n * sizeof(double);

    std::vector<double> h_A(n * n);
    std::vector<double> h_B(n * n);
    std::vector<double> h_C(n * n, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-10.0, 10.0);

    for (int i = 0; i < n * n; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_total));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start_kernel));

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop_kernel));

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    float kernel_ms = 0.0f, total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_total, stop_total));

    // Расчет метрик производительности
    long long totalElements = static_cast<long long>(n) * n;
    long long vramBytes = totalElements * 3 * sizeof(double);
    double vramMegabytes = static_cast<double>(vramBytes) / (1024.0 * 1024.0);
    long long totalOps = 2 * static_cast<long long>(n) * n * n;
    
    double kernelSeconds = kernel_ms / 1000.0;
    double totalSeconds = total_ms / 1000.0;

    double kernelGflops = (kernelSeconds > 0.0) ? ((static_cast<double>(totalOps) / kernelSeconds) / 1e9) : 0.0;
    double totalGflops = (totalSeconds > 0.0) ? ((static_cast<double>(totalOps) / totalSeconds) / 1e9) : 0.0;

    std::ofstream reportFile("performance_report_cuda.txt", std::ios_base::app);
    std::ostream* outputs[2] = { &std::cout, reportFile.is_open() ? &reportFile : nullptr };

    for (int idx = 0; idx < 2; ++idx) {
        if (!outputs[idx]) continue;
        std::ostream& out = *outputs[idx];

        out << "\n========================================\n";
        out << "CUDA ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ (" << n << "x" << n << "):\n";
        out << "Сетка блоков (Grid): " << numBlocks.x << " x " << numBlocks.y 
            << " (Всего блоков: " << numBlocks.x * numBlocks.y << ")\n";
        out << "Размер блока (Block): " << threadsPerBlock.x << " x " << threadsPerBlock.y 
            << " (" << threadsPerBlock.x * threadsPerBlock.y << " потоков/блок)\n";
        out << "========================================\n";
        out << "1. Математическая сложность и Скорость:\n";
        out << "   - Всего элементов в одной матрице: " << totalElements << "\n";
        out << "   - Всего операций (FLOP): ~" << totalOps << " оп.\n";
        out << std::fixed << std::setprecision(4);
        out << "   - Время чистого расчета (Kernel): " << kernel_ms << " мс (" << kernelSeconds << " сек)\n";
        out << "   - Общее время (с передачей PCIe): " << total_ms << " мс (" << totalSeconds << " сек)\n";
        out << std::setprecision(2);
        out << "   - ПРОИЗВОДИТЕЛЬНОСТЬ (Kernel): " << kernelGflops << " GFLOPS\n";
        out << "   - ПРОИЗВОДИТЕЛЬНОСТЬ (Общая):  " << totalGflops << " GFLOPS\n";
        out << "\n2. Затраты видеопамяти (VRAM):\n";
        out << "   - Суммарная память под 3 матрицы: " << vramMegabytes << " МБ\n";
        out << "========================================\n";
    }

    if (reportFile.is_open()) reportFile.close();

    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
