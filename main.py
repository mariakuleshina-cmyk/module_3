import sqlite3
import numpy as np
from tensorflow.keras.models import load_model

DB_NAME = "data.db"
MODEL_NAME = "traffic_model.keras"


def load_trained_model():
    print(f"Загрузка модели из {MODEL_NAME}")
    model = load_model(MODEL_NAME)
    print("Модель загружена")
    return model


def create_predictions_table():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        temperature INTEGER, cars INTEGER, light INTEGER,
        prediction REAL, result TEXT)""")
    conn.commit()
    conn.close()


def save_prediction(temperature, cars, light, prediction, result):
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "INSERT INTO predictions (temperature, cars, light, prediction, result) VALUES (?,?,?,?,?)",
        (temperature, cars, light, float(prediction), result))
    conn.commit()
    conn.close()


def predict(model, temperature, cars, light):
    temp_norm = (temperature + 30) / 70.0
    cars_norm = cars / 80.0
    X = np.array([[temp_norm, cars_norm, light]])
    prob = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(prob))
    return idx + 1, float(prob[idx])


def traffic_level(cars, light):
    score = cars + (10 if not light else 0)
    return 0 if score < 25 else (1 if score < 55 else 2)


def add_to_traffic():
    try:
        t = int(input("Температура (-30..40): ").strip())
        c = int(input("Машин (0..80): ").strip())
        l = int(input("Свет (0-ночь, 1-день): ").strip())
    except ValueError:
        print("Ошибка: целые числа\n")
        return

    level = traffic_level(c, l)

    conn = sqlite3.connect(DB_NAME)
    conn.execute("""CREATE TABLE IF NOT EXISTS traffic (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        temperature INTEGER, cars INTEGER, light INTEGER, traffic_level INTEGER)""")
    conn.execute(
        "INSERT INTO traffic (temperature, cars, light, traffic_level) VALUES (?,?,?,?)",
        (t, c, l, level))
    conn.commit()
    conn.close()
    print(f"Добавлено: t={t}, cars={c}, light={l}, level={level}\n")


def run_prediction():
    model = load_trained_model()
    create_predictions_table()

    while True:
        t_in = input("Температура (q — выход): ").strip()
        if t_in.lower() == "q":
            break

        try:
            t = int(t_in)
            c = int(input("Машин (0-80): ").strip())
            l = int(input("Свет (0-ночь, 1-день): ").strip())
        except ValueError:
            print("Ошибка: целые числа\n")
            continue

        if not (-30 <= t <= 40) or not (0 <= c <= 80) or l not in (0, 1):
            print("Ошибка: значения вне диапазона\n")
            continue

        result_number, conf = predict(model, t, c, l)
        text = {1: "1 (свободно)", 2: "2 (средне)", 3: "3 (пробка)"}[result_number]
        light_str = "день" if l else "ночь"
        print(f"t={t}, машин={c}, {light_str} → {text} (уверенность {conf:.4f})\n")

        save_prediction(t, c, l, conf, text)

    print("Результаты сохранены в БД")


if __name__ == "__main__":
    while True:
        print("1 — добавить данные в traffic")
        print("2 — предсказать")
        print("0 — выход")
        choice = input("Выбор: ").strip()

        if choice == "1":
            add_to_traffic()
        elif choice == "2":
            run_prediction()
        elif choice == "0":
            break
        else:
            print("Неверный выбор\n")