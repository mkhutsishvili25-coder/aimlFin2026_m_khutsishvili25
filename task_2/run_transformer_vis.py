import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create figures directory
out_dir = os.path.join("task_2", "figures")
os.makedirs(out_dir, exist_ok=True)

# ----------------------------
# 1) Attention Visualization
# ----------------------------
np.random.seed(42)
tokens = ["login", "verify", "account", "secure", "now"]

attention_matrix = np.random.rand(len(tokens), len(tokens))
attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,5))
sns.heatmap(attention_matrix, annot=True, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
plt.title("Self-Attention Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "attention.png"), dpi=200)
plt.close()

# ----------------------------
# 2) Positional Encoding Visualization
# ----------------------------
def positional_encoding(position, d_model):
    pe = np.zeros((position, d_model))
    for pos in range(position):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe

pe = positional_encoding(50, 32)

plt.figure(figsize=(8,5))
plt.imshow(pe, aspect='auto', cmap='coolwarm')
plt.colorbar()
plt.title("Positional Encoding")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "positional_encoding.png"), dpi=200)
plt.close()

print("Visualizations saved in task_2/figures/")
