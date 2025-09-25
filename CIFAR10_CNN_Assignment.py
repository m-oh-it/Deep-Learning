
# === Imports & Setup ===
import os, random, numpy as np
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models

# Metrics & evaluation
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import pandas as pd

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



# === Helper utilities ===

# CIFAR-10 class names (index -> label)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def show_samples(images, labels, n=5, title="Sample images"):
    plt.figure(figsize=(12, 3))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        label_idx = int(labels[i])
        ax.set_title(CLASS_NAMES[label_idx])
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(6,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

    # Plot loss
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7,7))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



# === Load CIFAR-10 ===
from tensorflow.keras.datasets import cifar10
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = cifar10.load_data()

print("Original training data:", x_train_orig.shape, y_train_orig.shape)
print("Original test data:", x_test_orig.shape, y_test_orig.shape)

# Combine and re-split to exactly 80/20
X_all = np.concatenate([x_train_orig, x_test_orig], axis=0)
y_all = np.concatenate([y_train_orig, y_test_orig], axis=0).reshape(-1)

num_samples = X_all.shape[0]
idx = np.arange(num_samples)
np.random.shuffle(idx)

train_size = int(0.8 * num_samples)
train_idx, test_idx = idx[:train_size], idx[train_size:]

x_train, y_train = X_all[train_idx], y_all[train_idx]
x_test, y_test = X_all[test_idx], y_all[test_idx]

print("Training data shape:", x_train.shape, y_train.shape)
print("Test data shape:", x_test.shape, y_test.shape)
print("Unique labels:", len(np.unique(y_all)), np.unique(y_all))

# Show 5 sample images
show_samples(x_train, y_train, n=5, title="Task 1: Sample images & labels")

# Normalize to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0



# === Define CNN model ===
def build_cnn(input_shape=(32,32,3), num_classes=10, dropout_rate=0.25):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_rate),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

base_model = build_cnn()
base_model.summary()



# === Compile & Train ===
EPOCHS = 12  # adjust 10–20 as required
BATCH_SIZE = 64

base_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

history = base_model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    shuffle=True,
    verbose=1
)

# Plot training curves
plot_training_history(history)



# === Evaluate on test set ===
test_loss, test_acc = base_model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Predictions
y_prob = base_model.predict(x_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, CLASS_NAMES, normalize=False, title="Confusion Matrix (Counts)")
plot_confusion_matrix(cm, CLASS_NAMES, normalize=True, title="Confusion Matrix (Normalized)")

# Show examples of correct & incorrect predictions
def show_examples(images, true_labels, pred_labels, correct=True, k=10, title="Examples"):
    mask = (true_labels == pred_labels) if correct else (true_labels != pred_labels)
    idxs = np.where(mask)[0][:k]
    if len(idxs) == 0:
        print("No examples to show.")
        return
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(idxs):
        ax = plt.subplot(1, min(k, len(idxs)), i+1)
        plt.imshow(images[idx])
        ax.set_title(f"T:{CLASS_NAMES[int(true_labels[idx])]}\nP:{CLASS_NAMES[int(pred_labels[idx])]}",
                     fontsize=9)
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

show_examples(x_test, y_test, y_pred, correct=True, k=10, title="Correctly Classified")
show_examples(x_test, y_test, y_pred, correct=False, k=10, title="Incorrectly Classified")



# === Optimizer experiments ===
import pandas as pd

def train_with_optimizer(opt_name, build_fn, epochs=6):
    model = build_fn()
    if opt_name.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    elif opt_name.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
    elif opt_name.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam()
    else:
        raise ValueError("Unknown optimizer")

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    _ = model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2, verbose=1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return test_acc

results = pd.DataFrame({
    'Optimizer': ['Adam', 'SGD (momentum=0.9)', 'RMSprop'],
    'Test Accuracy': [
        train_with_optimizer('adam', lambda: build_cnn(), epochs=6),
        train_with_optimizer('sgd', lambda: build_cnn(), epochs=6),
        train_with_optimizer('rmsprop', lambda: build_cnn(), epochs=6)
    ]
})
print(results)
