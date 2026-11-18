import re
import pandas as pd
import matplotlib.pyplot as plt

with open("performance_report_cuda.txt", "r", encoding="utf-8") as f:
    text = f.read()

pattern = re.compile(
    r"CUDA ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ \((\d+)x\d+\):.*?"
    r"Размер блока \(Block\): (\d+) x \d+ \((\d+) потоков/блок\).*?"
    r"Время чистого расчета \(Kernel\): ([\d.]+) мс.*?"
    r"Общее время \(с передачей PCIe\): ([\d.]+) мс.*?"
    r"ПРОИЗВОДИТЕЛЬНОСТЬ \(Kernel\): ([\d.]+) GFLOPS.*?"
    r"ПРОИЗВОДИТЕЛЬНОСТЬ \(Общая\):\s+([\d.]+) GFLOPS",
    re.DOTALL
)

records = []
for match in pattern.finditer(text):
    records.append({
        "N": int(match.group(1)),
        "Block": f"{match.group(2)}x{match.group(2)} ({match.group(3)} пот.)",
        "BlockSize": int(match.group(2)),
        "Kernel_ms": float(match.group(4)),
        "Total_ms": float(match.group(5)),
        "Kernel_GFLOPS": float(match.group(6)),
        "Total_GFLOPS": float(match.group(7)),
    })

df = pd.DataFrame(records)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 300})

markers = {4: "o", 8: "s", 16: "^", 32: "D"}
colors = {4: "#d62728", 8: "#ff7f0e", 16: "#2ca02c", 32: "#1f77b4"}

# ---------------------------------------------------------
# ГРАФИК 1: Зависимость времени расчета (Kernel) от размера N
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
for b_size, group in df.groupby("BlockSize"):
    label = group["Block"].iloc[0]
    ax.plot(
        group["N"], group["Kernel_ms"],
        marker=markers[b_size], color=colors[b_size],
        linewidth=2.2, markersize=7, label=f"Блок {label}"
    )

ax.set_title("Время чистого расчета ядра (Kernel) в зависимости от размера матрицы", fontsize=13, pad=12)
ax.set_xlabel("Размер матрицы (N × N)")
ax.set_ylabel("Время выполнения, мс")
ax.set_xticks(sorted(df["N"].unique()))
ax.legend(title="Конфигурация блока")
plt.tight_layout()
plt.savefig("1_cuda_kernel_time.png")
plt.close()

# ---------------------------------------------------------
# ГРАФИК 2: Производительность (Kernel vs Общая GFLOPS)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

for b_size, group in df.groupby("BlockSize"):
    label = group["Block"].iloc[0]
    ax1.plot(
        group["N"], group["Kernel_GFLOPS"],
        marker=markers[b_size], color=colors[b_size],
        linewidth=2.2, markersize=7, label=f"Блок {label}"
    )
    ax2.plot(
        group["N"], group["Total_GFLOPS"],
        marker=markers[b_size], color=colors[b_size],
        linewidth=2.2, markersize=7, label=f"Блок {label}"
    )

ax1.set_title("Производительность ядра (Kernel GFLOPS)", fontsize=12)
ax1.set_xlabel("Размер матрицы (N × N)")
ax1.set_ylabel("Производительность, GFLOPS")
ax1.set_xticks(sorted(df["N"].unique()))
ax1.legend(title="Конфигурация блока")

ax2.set_title("Общая производительность с учетом PCIe (Total GFLOPS)", fontsize=12)
ax2.set_xlabel("Размер матрицы (N × N)")
ax2.set_xticks(sorted(df["N"].unique()))
ax2.legend(title="Конфигурация блока")

plt.suptitle("Сравнение реальной производительности GPU на разных сетках блоков", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("2_cuda_gflops_comparison.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# ГРАФИК 3: Доля накладных расходов на передачу данных (PCIe)
# ---------------------------------------------------------
df_16 = df[df["BlockSize"] == 16].copy()
df_16["PCIe_ms"] = df_16["Total_ms"] - df_16["Kernel_ms"]
df_16["PCIe_share_%"] = (df_16["PCIe_ms"] / df_16["Total_ms"]) * 100

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(df_16["N"].astype(str), df_16["PCIe_share_%"], color="#4c72b0", width=0.55)

for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4), textcoords="offset points",
        ha="center", va="bottom", fontweight="bold"
    )

ax.set_title("Доля времени на копирование памяти по шине PCIe (для блока 16×16)", fontsize=13, pad=12)
ax.set_xlabel("Размер матрицы (N × N)")
ax.set_ylabel("Доля затрат на PCIe от общего времени, %")
ax.set_ylim(0, max(df_16["PCIe_share_%"]) * 1.15)
plt.tight_layout()
plt.savefig("3_cuda_pcie_overhead.png")
plt.close()

print("Графики успешно сохранены: 1_cuda_kernel_time.png, 2_cuda_gflops_comparison.png, 3_cuda_pcie_overhead.png")

