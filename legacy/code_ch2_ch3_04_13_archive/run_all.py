"""
Запуск всех сценариев моделирования (Главы 2, 3, 4).
Выполнять из директории code_app/.
"""
import os
import sys
# Принудительно UTF-8 на Windows (cp1251 не поддерживает кириллицу в Python 3)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

import run_ch2
import run_ch2_path
import run_ch3
import run_ch4_line
import run_ch4_spiral

if __name__ == "__main__":
    print("=" * 60)
    print("Запуск всех сценариев диссертации")
    print("=" * 60)

    print("\n--- Глава 2: Стабилизация в точке (пример 1) ---")
    run_ch2.main()

    print("\n--- Глава 2: Движение по ломаной (пример 2) ---")
    run_ch2_path.main()

    print("\n--- Глава 3: Следящее управление (спираль) ---")
    run_ch3.main()

    print("\n--- Глава 4: Согласованное управление (прямая) ---")
    run_ch4_line.main()

    print("\n--- Глава 4: Согласованное управление (спираль) ---")
    run_ch4_spiral.main()

    print("\n" + "=" * 60)
    print("Все сценарии выполнены. Результаты в директории code_app/out_images/")
    print("=" * 60)
