import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import json

dataset_file = 'contract_labled_1.csv'
training_file = 'Training result.json'

#########################################Thống kê các lỗi trong dataset###############################################
def vuln_statistic_fig():
        df = pd.read_csv(dataset_file)
        labels = df[['Label', 'Address']]
        gbdf = labels.groupby(["Label"]).count().reset_index()


        name = {0: 'Clean',1: 'Arithmetic', 2: 'Reentrancy', 3: 'Time Manipulation', 4: 'TOD'}

        labels = [name[x] for x in gbdf[['Label']].values.flatten()]
        sizes = gbdf[['Address']].values.flatten()
        explode = (0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig, ax = plt.subplots(figsize=(20, 10), )
        patches, texts, autotexts = ax.pie(sizes, explode=explode, labels=sizes, autopct='%1.1f%%',
                shadow=False, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.legend(patches, labels,
                title="Vulnerabilites",
                loc="center left",
                bbox_to_anchor=(0.8, 0, 0.5, 1), prop={'size': 20})
        fig.set_facecolor('w')
        [ _.set_fontsize(20) for _ in texts ]
        [ _.set_fontsize(20) for _ in autotexts ]


################################################## Thống kê training #################################################################

def get_metric(data, n_gram, dataset, algorithms, metric):
        score_dataset = list()
        for algorithm in algorithms:
                score = list()
                for gram in n_gram:
                        score.append(data[gram][dataset][algorithm][metric])
                score_dataset.append(score)
        return score_dataset

with open(training_file) as f:
        training_result = json.load(f)

n_gram = list(training_result.keys())
dataset = list(training_result[n_gram[0]].keys())
algorithms = list(training_result[n_gram[0]][dataset[0]])
color = ['red', 'orange', '#CDCD00', 'green', 'blue', 'indigo', 'violet', 'cyan']

accuracy_dataset_0 = get_metric(training_result, n_gram, dataset[0], algorithms, 'accuracy')
precision_dataset_0 = get_metric(training_result, n_gram, dataset[0], algorithms, 'precision')
recall_dataset_0 = get_metric(training_result, n_gram, dataset[0], algorithms, 'recall')
f1_dataset_0 = get_metric(training_result, n_gram, dataset[0], algorithms, 'f1')
train_time_dataset_0 = get_metric(training_result, n_gram, dataset[0], algorithms, 'train time')

accuracy_dataset_1 = get_metric(training_result, n_gram, dataset[1], algorithms, 'accuracy')
precision_dataset_1 = get_metric(training_result, n_gram, dataset[1], algorithms, 'precision')
recall_dataset_1 = get_metric(training_result, n_gram, dataset[1], algorithms, 'recall')
f1_dataset_1 = get_metric(training_result, n_gram, dataset[1], algorithms, 'f1')
train_time_dataset_1 = get_metric(training_result, n_gram, dataset[1], algorithms, 'train time')

algorithms.extend([algorithm + ' 1' for algorithm in algorithms])
accuracy = accuracy_dataset_0 + accuracy_dataset_1
precision = precision_dataset_0 + precision_dataset_1
recall = recall_dataset_0 + recall_dataset_1
f1 = f1_dataset_0 + f1_dataset_1

#--------------------------------------------- Training time ---------------------------------------------#
def time_fig():
        fig = plt.figure()
        for i in range(4):
                ax = fig.add_subplot(2, 2, i + 1)
                bars = ax.barh(n_gram, train_time_dataset_0[i], color=color[i])
                ax.set_xlim(0, train_time_dataset_0[i][7] + train_time_dataset_0[i][3])
                ax.set_xlabel('Seconds')
                ax.set_title(algorithms[i])
                ax.bar_label(bars, labels=[round(x, 3) for x in bars.datavalues], label_type='edge', padding=5)

#--------------------------------------------- Accuracy ---------------------------------------------#
def accuracy_fig():
        fig, ax = plt.subplots()
        for i in range(8):
                ax.plot(n_gram, accuracy[i], color=color[i], marker='o', label=algorithms[i])

        ax.set_title('Compare Accuracy Score of N-grams between Classcifiers', fontsize=14)
        ax.set_xlabel('N-grams', fontsize=14)
        ax.set_ylabel('Accuracy Score', fontsize=14)
        ax.legend(loc='best')
        ax.set_yticks(np.arange(0, 1.02, step=0.05))

#--------------------------------------------- Precision ---------------------------------------------#
def precision_fig():
        fig, ax = plt.subplots()
        for i in range(8):
                ax.plot(n_gram, precision[i], color=color[i], marker='o', label=algorithms[i])

        ax.set_title('Compare Precision Score of N-grams between Classcifiers', fontsize=14)
        ax.set_xlabel('N-grams', fontsize=14)
        ax.set_ylabel('Precision Score', fontsize=14)
        ax.legend(loc='best')
        ax.set_yticks(np.arange(0, 1.02, step=0.05))

#--------------------------------------------- Recall ---------------------------------------------#
def recall_fig():
        fig, ax = plt.subplots()
        for i in range(8):
                ax.plot(n_gram, recall[i], color=color[i], marker='o', label=algorithms[i])

        ax.set_title('Compare Recall Score of N-grams between Classcifiers', fontsize=14)
        ax.set_xlabel('N-grams', fontsize=14)
        ax.set_ylabel('Recall Score', fontsize=14)
        ax.legend(loc='best')
        ax.set_yticks(np.arange(0, 1.02, step=0.05))

#--------------------------------------------- F1 ---------------------------------------------#
def f1_fig():
        fig, ax = plt.subplots()
        for i in range(8):
                ax.plot(n_gram, f1[i], color=color[i], marker='o', label=algorithms[i])

        ax.set_title('Compare F1 Score of N-grams between Classcifiers', fontsize=14)
        ax.set_xlabel('N-grams', fontsize=14)
        ax.set_ylabel('F1 Score', fontsize=14)
        ax.legend(loc='best')
        ax.set_yticks(np.arange(0, 1.02, step=0.05))

vuln_statistic_fig()
time_fig()
accuracy_fig()
precision_fig()
recall_fig()
f1_fig()
plt.show()

