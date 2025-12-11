# RL_PyTorch/main.py
"""
Главная точка входа системы.
Консольное меню, оркестрирующее все режимы:
- работа с master-историей
- снапшоты для обучения
- training режим
- live режим
- сервисные функции
"""

from market_data.market_data_layer import (
    update_master_from_live_logs,
    validate_master_history,
    create_snapshot_from_master,
    list_snapshots,
    validate_snapshot,
    delete_snapshot,
    download_dukascopy_history,
    build_master_from_external,
)


from market_data.providers.training_provider import run_training_loop_interactive, run_validation_loop
from market_data.providers.live_provider import start_live_trading, start_live_logger
from market_data.connectors.mt5_connector import (
    test_mt5_connection,
    show_account_info,
    list_symbols,
)


def main_menu() -> None:
    while True:
        print("\n=== MAIN MENU ===")
        print("1. История (master data)")
        print("2. Снапшоты (training data)")
        print("3. Training режим")
        print("4. Live режим")
        print("5. Сервис")
        print("0. Выход")

        choice = input("Выберите пункт меню: ").strip()

        if choice == "1":
            history_menu()
        elif choice == "2":
            snapshot_menu()
        elif choice == "3":
            training_menu()
        elif choice == "4":
            live_menu()
        elif choice == "5":
            service_menu()
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неверный выбор, попробуйте снова.")


# ---------- History (master data) ----------

def history_menu() -> None:
    while True:
        print("\n--- История (master data) ---")
        print("1. Скачать историю M1 с Dukascopy в external")
        print("2. Собрать master-историю из external (Dukascopy → master)")
        print("3. Проверить целостность master-истории")
        print("4. Обновить master-историю из live-логов")
        print("0. Назад")

        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            download_dukascopy_history()
        elif choice == "2":
            build_master_from_external()
        elif choice == "3":
            validate_master_history()
        elif choice == "4":
            update_master_from_live_logs()
        elif choice == "0":
            break
        else:
            print("Неверный выбор, попробуйте снова.")

# ---------- Snapshots ----------

def snapshot_menu() -> None:
    while True:
        print("\n--- Снапшоты (training data) ---")
        print("1. Создать snapshot из master-истории")
        print("2. Показать список snapshot-файлов")
        print("3. Проверить snapshot")
        print("4. Удалить snapshot")
        print("0. Назад")

        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            train_start = input("Введите дату начала (YYYY-MM-DD): ").strip()
            train_end = input("Введите дату конца   (YYYY-MM-DD): ").strip()

            print("Выберите тип snapshot:")
            print(" 1. train")
            print(" 2. valid")
            print(" 3. test")
            print(" 4. other")
            role_choice = input("Тип [по умолчанию: 1]: ").strip()

            role_map = {
                "1": "train",
                "2": "valid",
                "3": "test",
                "4": "other",
            }
            role = role_map.get(role_choice, "train")

            print(f"[Snapshot] Создание snapshot с {train_start} по {train_end}, тип={role}")
            create_snapshot_from_master(train_start, train_end, role)

        elif choice == "2":
            list_snapshots()

        elif choice == "3":
            validate_snapshot()

        elif choice == "4":
            delete_snapshot()

        elif choice == "0":
            break

        else:
            print("Неверный выбор, попробуйте снова.")

# ---------- Training ----------

def training_menu() -> None:
    while True:
        print("\n--- Training ---")
        print("1. Запустить обучение на snapshot")
        print("2. Тест/валидация модели на snapshot")  # пока можно скрыть / закомментировать
        print("0. Назад")
        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            run_training_loop_interactive()
        elif choice == "2":
            run_validation_loop()
        elif choice == "0":
            break
        else:
            print("Неверный выбор, попробуйте снова.")

# ---------- Live ----------

def live_menu() -> None:
    while True:
        print("\n--- Live режим ---")
        print("1. Запустить live-торговлю (модель + исполнение)")
        print("2. Запустить live-логгер (без торговли)")
        print("0. Назад")

        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            start_live_trading()
        elif choice == "2":
            start_live_logger()
        elif choice == "0":
            break
        else:
            print("Неверный выбор, попробуйте снова.")


# ---------- Service ----------

def service_menu() -> None:
    while True:
        print("\n--- Сервис ---")
        print("1. Проверить подключение к MT5")
        print("2. Показать информацию об аккаунте/сервере")
        print("3. Показать список символов")
        print("0. Назад")

        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            test_mt5_connection()
        elif choice == "2":
            show_account_info()
        elif choice == "3":
            list_symbols()
        elif choice == "0":
            break
        else:
            print("Неверный выбор, попробуйте снова.")


if __name__ == "__main__":
    main_menu()
