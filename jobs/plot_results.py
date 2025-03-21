import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
print(df)

guess_score = 0.1
group_idx = [[1,4], [5,7], [8,10], [11,16]]
test_accuracy = np.array([np.mean(df['test_accuracy'][i:j]) for i,j in group_idx])
val_accuracy = np.array([np.mean(df['val_accuracy'][i:j]) for i,j in group_idx])
kfold_accuracy = [0.62]

plt.plot(test_accuracy, color='blue')
plt.scatter(range(len(test_accuracy)), test_accuracy, marker='x', color='blue', label='Test')

plt.plot(val_accuracy, color='red')
plt.scatter(range(len(val_accuracy)), val_accuracy, marker='x', color='red', label='Val')

plt.scatter([len(test_accuracy)-1], kfold_accuracy, label='KFold', marker='x', color='green') # placeholder

plt.axhline(y=guess_score, color='grey', linestyle='--', alpha=0.2, label='Random')

plt.xlabel('Development Iteration')
plt.ylabel('Accuracy')
plt.title('Mean job score over development')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.xlim(0, len(test_accuracy)-0.9)
plt.savefig('results.png')
