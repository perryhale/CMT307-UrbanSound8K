import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load results
df = pd.read_csv('results.csv')
print(df)

# extract group scores
guess_score = 0.1
group_idx = [[1,4], [5,7], [8,10], [11,16]]
test_accuracy = np.array([np.mean(df['test_accuracy'][i:j]) for i,j in group_idx])
val_accuracy = np.array([np.mean(df['val_accuracy'][i:j]) for i,j in group_idx])
kfold_accuracy = [0.62] # placeholder
stage_axis = range(1, len(test_accuracy)+1)
stage_labels = ['constant lr', '+decay schedule', '+low init lr', '-high init lr \n +noise variations']

# initialise figure
fig, ax = plt.subplots(figsize=(7,5))

# draw train scores
ax.plot(stage_axis, test_accuracy, color='green')
ax.scatter(stage_axis, test_accuracy, marker='x', color='green', label='Test')

# draw val scores
ax.plot(stage_axis, val_accuracy, color='red')
ax.scatter(stage_axis, val_accuracy, marker='x', color='red', label='Validation')

# draw kfold scores
ax.scatter([stage_axis[-1]], kfold_accuracy, label='KFold', marker='x', color='blue')

# draw random threshold
ax.axhline(y=guess_score, color='grey', linestyle='--', alpha=0.2, label='Random')

# draw labels
ax.set_xlabel('Manual tuning stage')
ax.set_xticks(stage_axis, stage_labels, rotation=45, ha='right')
ax.set_ylabel('Average accuracy')
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.015))

# set limits
ax.set_ylim(0,1)
ax.set_xlim(1, stage_axis[-1]+0.1)

# save and close
fig.subplots_adjust(right=0.75, bottom=0.3)
plt.savefig(f'{__file__.replace(".py","")}-001.png')
plt.close()

# extract c* and d* model results
cd_star = pd.DataFrame(df.iloc[11:16,:])

# format labels
cd_star['label'] = cd_star['pretrain_noise'] + ' - ' + cd_star['pretrain_data']

# sort by mean score
cd_star['mean_score'] = cd_star[['val_accuracy', 'test_accuracy']].mean(axis=1)
cd_star = cd_star.sort_values(by='mean_score', ascending=False).drop(columns=['mean_score'])

# initialise figure
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cd_star))
width = 0.35

# draw plots
bars1 = ax.bar(x - width/2, cd_star['val_accuracy'], width, label='Validation', color='red')
bars2 = ax.bar(x + width/2, cd_star['test_accuracy'], width, label='Test', color='green')

# draw labels
ax.set_xlabel('Fine-tune from [noise type]-[dataset] denoising pretrain')
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(cd_star['label'], rotation=45, ha='right')
ax.legend()

# set limits
ax.set_ylim(0.5,0.75)

# save and close
plt.tight_layout()
plt.savefig(f'{__file__.replace(".py","")}-002.png')
plt.close()
