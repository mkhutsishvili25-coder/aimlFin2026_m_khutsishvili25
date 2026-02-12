import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------------
# 1) Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# 2) Embedded cybersecurity dataset (toy)
#    0 = benign, 1 = phishing
# ----------------------------
samples = [
    # benign
    ("google.com", 0),
    ("nku.edu", 0),
    ("amazon.com", 0),
    ("github.com", 0),
    ("microsoft.com", 0),
    ("openai.com", 0),
    ("wikipedia.org", 0),
    ("bbc.co.uk", 0),
    ("paypal.com", 0),
    ("apple.com", 0),
    ("stackoverflow.com", 0),
    ("nytimes.com", 0),
    ("irs.gov", 0),
    ("revenue.gov", 0),
    ("bankofgeorgia.ge", 0),

    # phishing-like / suspicious
    ("paypaI-login-secure.com", 1),     # "I" instead of "l"
    ("micros0ft-support-login.net", 1),  # zero in microsoft
    ("goog1e-verify-account.com", 1),    # 1 instead of l
    ("appleid-confirm-security.com", 1),
    ("secure-paypal-update.info", 1),
    ("account-verify-banking-login.com", 1),
    ("signin-microsoft-security-alert.com", 1),
    ("github-security-check.com", 1),
    ("amazon-billing-verify.com", 1),
    ("update-payment-method-now.com", 1),
    ("revenue-refund-claim.com", 1),
    ("bank-login-confirmation.net", 1),
    ("verify-identity-now-support.com", 1),
    ("support-secure-login-alert.com", 1),
    ("paypa1-account-l0gin.com", 1),
]

texts = [s[0].lower() for s in samples]
labels = np.array([s[1] for s in samples], dtype=np.int32)

# ----------------------------
# 3) Character vocabulary
#    We keep it simple: lowercase letters, digits, and common URL symbols.
# ----------------------------
alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789.-_")
char_to_id = {c: i + 1 for i, c in enumerate(alphabet)}  # 0 reserved for padding

max_len = 40  # fixed length (pad or truncate)

def encode(text: str) -> np.ndarray:
    text = text.lower()
    ids = [char_to_id.get(ch, 0) for ch in text]  # unknown -> 0
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)

X = np.stack([encode(t) for t in texts], axis=0)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, labels, test_size=0.3, random_state=SEED, stratify=labels
)

# ----------------------------
# 4) 1D CNN model (character-level)
# ----------------------------
vocab_size = len(alphabet) + 1  # + padding(0)
embed_dim = 16

model = models.Sequential([
    layers.Input(shape=(max_len,)),
    layers.Embedding(input_dim=vocab_size, output_dim=embed_dim),

    layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling1D(pool_size=2),

    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")  # binary classification
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# ----------------------------
# 5) Train
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=8,
    verbose=0
)

# ----------------------------
# 6) Evaluation
# ----------------------------
val_probs = model.predict(X_val, verbose=0).ravel()
val_pred = (val_probs >= 0.5).astype(np.int32)

print("\nClassification report:")
print(classification_report(y_val, val_pred, digits=3))

cm = confusion_matrix(y_val, val_pred)
print("Confusion matrix:\n", cm)

# ----------------------------
# 7) Visualizations (saved to figures/)
# ----------------------------
out_dir = os.path.join("task_1", "figures")
os.makedirs(out_dir, exist_ok=True)

# Accuracy/Loss plot
plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "accuracy.png"), dpi=200)
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "loss.png"), dpi=200)
plt.close()

# Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign(0)", "phishing(1)"])
fig, ax = plt.subplots()
disp.plot(ax=ax, values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
plt.close()

print(f"\nSaved figures to: {out_dir}")

# ----------------------------
# 8) Quick manual test
# ----------------------------
test_domains = [
    "paypal.com",
    "paypaI-login-secure.com",
    "mybank-login-confirm.net",
    "github.com"
]

for d in test_domains:
    p = model.predict(encode(d).reshape(1, -1), verbose=0).ravel()[0]
    print(f"{d:30s} -> phishing probability = {p:.3f}")