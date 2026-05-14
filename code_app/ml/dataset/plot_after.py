import sys

sys.path.insert(0, "code_app")
import os

os.chdir("code_app")

from scenarios.run_build_dataset import plot_dataset_stats, print_csv_summary

CSV = "ml/data/dataset_large.csv"
IMGS = "out_images/dataset_large"

print_csv_summary(CSV)  # статистика в терминал
plot_dataset_stats(CSV, IMGS)  # 3 png-графика в IMGS/