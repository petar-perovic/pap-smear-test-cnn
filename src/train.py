from src.dataset import get_train_test_data
from src.model import create_model
from tensorflow.keras.callbacks import EarlyStopping

X_train, X_test, y_train, y_test = get_train_test_data()

model = create_model(X_train.shape[1:])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    callbacks=[early_stop]
)

model.save("results/cnn_model.h5")
