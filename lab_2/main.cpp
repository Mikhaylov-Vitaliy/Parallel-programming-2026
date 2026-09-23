#include <iostream>
#include <vector>
#include <print>
#include <fstream>
#include <chrono>
#include <random>
#include <string>
#include <filesystem>
#include <omp.h>

using Matrix = std::vector<std::vector<double>>;
namespace fs = std::filesystem;

void generateMatrixFile(const std::string& filename, int n) {
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::println(std::cerr, "Ошибка: Не удалось создать файл для генерации: {}", filename);
		return;
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-10.0, 10.0);

	std::println(file, "{}", n);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			std::print(file, "{:.2f} ", dis(gen));
		}
		std::println(file);
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			std::print(file, "{:.2f} ", dis(gen));
		}
		std::println(file);
	}

	file.close();
	std::println("Генерация успешно завершена. Создан файл {} (Размер {}x{})", filename, n, n);
}

Matrix multSquareMatrix(const Matrix& A, const Matrix& B, int n) {
	Matrix C(n, std::vector<double>(n, 0.0));

#pragma omp parallel for default(none) shared(A, B, C, n)  // - Новое
	for (int i = 0; i < n; ++i) {
		for (int k = 0; k < n; ++k) {
			for (int j = 0; j < n; ++j) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return C;
}

int main() {
	const std::string fileGenerated = "matrix_generated.txt";
	const std::string fileManual = "matrix.txt";
	std::string targetFile = "";

	std::println("1. Сгенерировать случайную матрицу (до 2000х2000)");
	std::println("2. Использовать матрицу из файла (matrix.txt)");
	std::print("Выберите вариант: ");

	int choice = 0;
	if (!(std::cin >> choice)) {
		std::println(std::cerr, "Ошибка ввода: введено не число!");
		return 1;
	}

	if (choice == 1) {
		std::print("Введите размер матрицы (N): ");
		int customSize = 0;
		if (!(std::cin >> customSize) || customSize <= 0 || customSize > 1000) {
			std::println(std::cerr, "Ошибка: Некорректный размер матрицы!");
			return 1;
		}
		generateMatrixFile(fileGenerated, customSize);
		targetFile = fileGenerated;
	}
	else if (choice == 2) {
		targetFile = fileManual;
	}
	else {
		std::println(std::cerr, "Ошибка: Неверный пункт меню!");
		return 1;
	}

	std::print("Введите количество потоков OpenMP: ");
	int threads_count = 1;
	if (!(std::cin >> threads_count) || threads_count <= 0) {
		std::println(std::cerr, "Ошибка: Некорректное число потоков!");
		return 1;
	}
	omp_set_num_threads(threads_count); // Установка количества потоков для OpenMP

	std::ifstream file(targetFile);
	if (!file.is_open()) {
		std::println(std::cerr, "Ошибка: Не удалось открыть файл {}", targetFile);
		return 1;
	}

	int n = 0;
	file >> n;

	if (n <= 0) {
		std::cerr << "Ошибка: Некорректный размер матрицы в файле." << std::endl;
		return 1;
	}

	Matrix A(n, std::vector<double>(n));
	Matrix B(n, std::vector<double>(n));

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			file >> A[i][j];
		}
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			file >> B[i][j];
		}
	}

	file.close();

	auto start = std::chrono::high_resolution_clock::now();

	Matrix C = multSquareMatrix(A, B, n);

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;

	std::ofstream outfile("result.txt");

	if (!outfile.is_open()) {
		std::cerr << "Ошибка создания файла result.txt" << std::endl;
		return 1;
	}

	std::println(outfile, "Результат умножения матриц из файла (Размер {} x {}):", n, n);
	std::println(outfile, "Время расчета: {:.4f} мс\n", duration.count());

	for (const auto& row : C) {
		for (double a : row) {
			std::print(outfile, "{:.2f}\t", a);
		}
		std::println(outfile);
	}

	outfile.close();

	long long totalElements = static_cast<long long>(n) * n;
	long long ramBytes = totalElements * 3 * sizeof(double);
	double ramMegabytes = static_cast<double>(ramBytes) / (1024.0 * 1024.0);

	long long totalOps = 2 * static_cast<long long>(n) * n * n;
	double seconds = duration.count() / 1000.0;

	double gflops = 0.0;
	if (seconds > 0.0) {
		gflops = (static_cast<double>(totalOps) / seconds) / 1'000'000'000.0;
	}

	double inputFileSizeKB = 0.0;
	double outputFileSizeKB = 0.0;
	if (fs::exists(targetFile)) {
		inputFileSizeKB = static_cast<double>(fs::file_size(targetFile)) / 1024.0;
	}
	if (fs::exists("result.txt")) {
		outputFileSizeKB = static_cast<double>(fs::file_size("result.txt")) / 1024.0;
	}

	std::println("\n========================================");
	std::println("ОТЧЕТ ОБ ОБЪЁМЕ И ПРОИЗВОДИТЕЛЬНОСТИ ({}x{}):", n, n);
	std::println("========================================");
	std::println("Активных потоков OpenMP: {}", threads_count); // Новая строка
	std::println("1. Математическая сложность и Скорость:");
	std::println("   - Всего элементов в одной матрице: {}", totalElements);
	std::println("   - Всего операций (FLOP): ~{} оп.", totalOps);
	std::println("   - Время расчета алгоритма: {:.4f} мс ({:.6f} сек)", duration.count(), seconds);
	std::println("   - ПРОИЗВОДИТЕЛЬНОСТЬ: {:.2f} GFLOPS", gflops);

	std::println("\n2. Затраты памяти (ОЗУ):");
	std::println("   - Память под 3 матрицы в программе: {:.2f} МБ", ramMegabytes);

	std::println("\n3. Объём данных на диске:");
	std::println("   - Размер исходного файла ({}): {:.2f} КБ", targetFile, inputFileSizeKB);
	std::println("   - Размер файла с результатом (result.txt): {:.2f} КБ", outputFileSizeKB);
	std::println("========================================");

	return 0;
}