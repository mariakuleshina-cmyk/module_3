import sqlite3
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

DB_NAME = "data.db"
MODEL_NAME = "traffic_model.keras"


def load_data():
    """Загрузка данных из БД и подготовка признаков и меток."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT temperature, cars, light, traffic_level FROM traffic")
    rows = cursor.fetchall()
    conn.close()

    X = np.array([[row[0], row[1], row[2]] for row in rows], dtype=float)

    Y = np.array([row[3] for row in rows], dtype=int)

    X[:, 0] = (X[:, 0] + 30) / 70.0   
    X[:, 1] = X[:, 1] / 80.0          

    Y = to_categorical(Y, num_classes=3)

    return X, Y


def create_model():
    """Создание и компиляция нейросети."""
    model = Sequential([
        Dense(16, activation='relu', input_shape=(3,)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def train_model():
    """Основной процесс обучения."""
    print("Загрузка данных из БД...")
    X, Y = load_data()
    print(f"Загружено {len(X)} записей")

    print("Создание модели...")
    model = create_model()

    print("Обучение...")
    history = model.fit(X, Y, epochs=100, verbose=1)

    print(f"Точность: {history.history['accuracy'][-1]:.4f}")
    print(f"Ошибка: {history.history['loss'][-1]:.4f}")

    model.save(MODEL_NAME)
    print(f"Модель сохранена в {MODEL_NAME}")

    return model


if __name__ == "__main__":
    train_model()