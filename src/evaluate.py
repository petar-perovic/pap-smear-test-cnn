import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from src.dataset import get_train_test_data


def main():
    X_train, X_test, y_train, y_test = get_train_test_data()

    model = load_model("results/cnn_model.keras")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "LG lesion", "HG lesion"]
    ))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "LG", "HG"],
        yticklabels=["Normal", "LG", "HG"]
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
