# Задание 4 - Адаптивная резонансная теория, "переосмысление" категоризации, категориального распознавания и запоминания
# https://github.com/NiklasMelton/AdaptiveResonanceLib
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from collections import Counter, defaultdict
from artlib import ART1, FuzzyART


def load_and_prepare_data(binarize=True, binarization_threshold=128, num_train=5000, num_test=1000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:num_train]
    y_train = y_train[:num_train]
    x_test = x_test[:num_test]
    y_test = y_test[:num_test]

    x_train = x_train.reshape((num_train, 28 * 28))
    x_test = x_test.reshape((num_test, 28 * 28))

    if binarize:
        x_train_new = (x_train >= binarization_threshold).astype(int)
        x_test_new = (x_test >= binarization_threshold).astype(int)
    else:
        x_train_new = x_train
        x_test_new = x_test

    return x_train_new, y_train, x_test_new, y_test


def run(rho, x_tr_og, y_tr, x_te_og, y_te, model, was_data_binarized):
    if was_data_binarized:
        lower = np.zeros(x_tr_og.shape[1], dtype=int)
        upper = np.ones(x_te_og.shape[1], dtype=int)
    else:
        lower = np.array([0.] * x_tr_og.shape[1])
        upper = np.array([255.] * x_te_og.shape[1])
    model.set_data_bounds(lower, upper)

    model.fit(x_tr_og, verbose=True)

    predicted_labels = model.predict(x_te_og)

    cat2digits = defaultdict(list)
    for lbl, digit in zip(predicted_labels, y_te):
        cat2digits[lbl].append(digit)
    cat2major = {cat: Counter(digs).most_common(1)[0][0] for cat, digs in cat2digits.items()}

    correct = sum(1 for lbl, digit in zip(predicted_labels, y_te) if cat2major.get(lbl) == digit)
    accuracy = correct / len(y_te)

    print(f"Количество категорий: {len(cat2major)}")
    print(f"Оценка точности кластеризации: {accuracy:.4f}")
    print("Категории и цифры:")
    for cat, major in list(cat2major.items())[:]:
        print(f"  Категория {cat} → цифра {major} (кол-во: {len(cat2digits[cat])})")

    return {
        "predicted_labels": predicted_labels,
        "rho": rho,
        "num_categories": len(cat2major),
        "accuracy": accuracy,
        "cat2major": cat2major
    }


if __name__ == "__main__":
    is_art1 = bool(input("Использовать ART1? 1 - да, 0 - нет "))
    if is_art1:
        model = ART1(rho=0.7, L=1.0)
    else:
        model = FuzzyART(rho=0.7, alpha=0.0, beta=1.0)

    x_train_bin, y_train, x_test_bin, y_test = load_and_prepare_data(
        binarize=is_art1,
        binarization_threshold=128,
        num_train=3000,
        num_test=1000
    )

    results = []
    for rho in [0.7]:
        res = run(rho, x_train_bin, y_train, x_test_bin, y_test, model, is_art1)
        results.append(res)

    print("=== Итог ===")
    for r in results:
        print(f"Ро={r['rho']}: Категории={r['num_categories']}, Точность≈{r['accuracy']:.4f}")
    model.visualize(x_test_bin, res['predicted_labels'])
    plt.show()
