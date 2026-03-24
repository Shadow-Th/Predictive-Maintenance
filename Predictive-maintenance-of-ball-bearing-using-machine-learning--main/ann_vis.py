import matplotlib.pyplot as plt
import numpy as np
accuracy = 0.9369
precision = [0.99, 0.67, 0.95]
recall    = [1.00, 0.75, 0.92]
f1_score  = [1.00, 0.71, 0.94]
classes   = ['Class 0', 'Class 1', 'Class 2']

x = np.arange(len(classes))
width = 0.2

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].bar(x - width, precision, width, label='Precision')
axes[0].bar(x, recall, width, label='Recall')
axes[0].bar(x + width, f1_score, width, label='F1-score')

axes[0].set_xticks(x)
axes[0].set_xticklabels(classes)
axes[0].set_ylabel('Scores')
axes[0].set_title('ANN Metrics per Class')
axes[0].set_ylim([0, 1.1])
axes[0].legend()
for i in range(len(classes)):
    axes[0].text(x[i] - width, precision[i] + 0.02, f"{precision[i]:.2f}", ha='center')
    axes[0].text(x[i], recall[i] + 0.02, f"{recall[i]:.2f}", ha='center')
    axes[0].text(x[i] + width, f1_score[i] + 0.02, f"{f1_score[i]:.2f}", ha='center')

axes[1].bar(['Accuracy'], [accuracy], color='mediumorchid')
axes[1].set_ylim([0, 1])
axes[1].set_title('ANN Test Accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].text(0, accuracy + 0.02, f"{accuracy:.4f}", ha='center')

plt.tight_layout()
plt.show()
