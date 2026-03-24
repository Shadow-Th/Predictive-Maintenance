import matplotlib.pyplot as plt
import numpy as np
accuracy = 0.9498
classes = ['Class 0', 'Class 1', 'Class 2']
precision = [1.00, 0.80, 0.93]
recall = [1.00, 0.67, 0.96]
f1_score = [1.00, 0.73, 0.95]
x = np.arange(len(classes))
width = 0.25 
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-score')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1-score by Class')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.ylim([0, 1.1])
plt.show()
plt.figure(figsize=(4, 6))
plt.bar(['Accuracy'], [accuracy], color='skyblue')
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.title('Overall Test Accuracy')
plt.text(0, accuracy + 0.02, f'{accuracy:.4f}', ha='center')
plt.show()
