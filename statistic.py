import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import json
from imblearn.over_sampling import SMOTE  

dataset_file = "csv/contract_labled_1_1-gram.csv"
training_file = 'Training result.json'
training_noSMOTE_file = 'Training result no Smote.json'
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

#########################################Thống kê các lỗi dùng SMOTE###############################################
def compare_SMOTE_fig():
        def count(count_vuln, label, isSMOTE):
                for i in label:
                        if i == 0:
                                count_vuln["Arithmetic"][isSMOTE] += 1
                        elif i == 2:
                                count_vuln["Reentrancy"][isSMOTE] += 1
                        elif i == 3:
                                count_vuln["Time Manipulation"][isSMOTE] += 1
                        elif i == 4:
                                count_vuln["TOD"][isSMOTE] += 1
                        else:
                                count_vuln["Clean"][isSMOTE] += 1
                return count_vuln
        
        df = pd.read_csv(dataset_file)
        df = df.fillna(0)
        X = df.iloc[:,6:].values
        y = df[['Label']].values.flatten()
        
        count_vuln = {'Clean': [0, 0], 'Arithmetic': [0, 0], 'Reentrancy': [0, 0], 'Time Manipulation': [0, 0], 'TOD': [0, 0]}

        count_vuln = count(count_vuln, y, 0)

        sm = SMOTE(random_state = 2)
        _, y_res = sm.fit_resample(X, y)

        count_vuln = count(count_vuln, y_res, 1)

        labels = count_vuln.keys()
        no_SMOTE = [count_vuln[label][0] for label in labels]
        _SMOTE = [count_vuln[label][1] for label in labels]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, no_SMOTE, width, label='Not use SMOTE')
        rects2 = ax.bar(x + width/2, _SMOTE, width, label='Use SMOTE')

        # Add some text for labels, title and custom x-axis tick labels, etc.\
        ax.set_title('Compare the number of vulnerabilities between not use SMOTE and use SMOTE')
        ax.set_xticks(x, labels)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        print(_SMOTE)

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

with open(training_noSMOTE_file) as f:
        training_noSMOTE_result = json.load(f)

n_gram = list(training_result.keys())
dataset = list(training_result[n_gram[0]].keys())
algorithms = list(training_result[n_gram[0]][dataset[0]])
color = ['black', '#006400', 'navy', '#E6D0C3', 'red', '#FFD700', '#00FF00', '#00BFFF', '#FF00FF', 'brown']

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

algorithms_noSMOTE = list(training_noSMOTE_result[n_gram[0]][dataset[0]])
accuracy_noSMOTE_dataset_0 = get_metric(training_noSMOTE_result, n_gram, dataset[0], algorithms_noSMOTE, 'accuracy')
precision_noSMOTE_dataset_0 = get_metric(training_noSMOTE_result, n_gram, dataset[0], algorithms_noSMOTE, 'precision')
recall_noSMOTE_dataset_0 = get_metric(training_noSMOTE_result, n_gram, dataset[0], algorithms_noSMOTE, 'recall')
f1_noSMOTE_dataset_0 = get_metric(training_noSMOTE_result, n_gram, dataset[0], algorithms_noSMOTE, 'f1')
train_time_noSMOTE_dataset_0 = get_metric(training_noSMOTE_result, n_gram, dataset[0], algorithms_noSMOTE, 'train time')

accuracy_noSMOTE_dataset_1 = get_metric(training_noSMOTE_result, n_gram, dataset[1], algorithms_noSMOTE, 'accuracy')
precision_noSMOTE_dataset_1 = get_metric(training_noSMOTE_result, n_gram, dataset[1], algorithms_noSMOTE, 'precision')
recall_noSMOTE_dataset_1 = get_metric(training_noSMOTE_result, n_gram, dataset[1], algorithms_noSMOTE, 'recall')
f1_noSMOTE_dataset_1 = get_metric(training_noSMOTE_result, n_gram, dataset[1], algorithms_noSMOTE, 'f1')
train_time_noSMOTE_dataset_1 = get_metric(training_noSMOTE_result, n_gram, dataset[1], algorithms_noSMOTE, 'train time')

algorithms_noSMOTE.extend([algorithms_noSMOTE + ' 1' for algorithms_noSMOTE in algorithms_noSMOTE])


#--------------------------------------------- Training time ---------------------------------------------#
def time_fig():
        fig = plt.figure()
        train_time = [i for i in train_time_dataset_0]
        train_time.append(train_time_dataset_1[-1])
        title = [algorithms[i] for i in range(5)]
        title.append("DNN 1")
        for i in range(int(len(algorithms)/2) + 1):
                ax = fig.add_subplot(3, 2, i + 1)
                bars = ax.barh(n_gram, train_time[i], color=color[i])
                ax.set_xlim(0, train_time[i][7] + train_time[i][3] + train_time[i][1])
                ax.set_xlabel('Seconds')
                ax.set_title(title[i])
                ax.bar_label(bars, labels=[round(x, 3) for x in bars.datavalues], label_type='edge', padding=5)

#--------------------------------------------- Training time ---------------------------------------------#
def time_noSMOTE_fig():
        fig = plt.figure()
        train_time = [i for i in train_time_dataset_0]
        train_time_noSMOTE = [i for i in train_time_noSMOTE_dataset_0]
        title = [algorithms_noSMOTE[i] for i in range(4)]
        for i in range(int(len(algorithms_noSMOTE)/2)):
                ax = fig.add_subplot(2, 2, i + 1)
                ind = np.arange(len(n_gram))
                width = 0.4
                l1 = ax.barh(ind-0.2, train_time[i], width, label = 'Use SMOTE', color='red')
                l2 = ax.barh(ind+0.2, train_time_noSMOTE[i], width, label = 'No SMOTE', color='blue')
                ax.set_yticks(ind, n_gram)
                ax.set_xlim(0, max(train_time[i] + train_time_noSMOTE[i]) + train_time[i][2])
                ax.set_xlabel('Seconds')
                ax.set_title(title[i])
                ax.legend(loc='best')
                #ax.bar_label(bars, labels=[round(x, 3) for x in bars.datavalues], label_type='edge', padding=5)

#--------------------------------------------- Accuracy ---------------------------------------------#
def accuracy_fig():
        fig, ax = plt.subplots()
        for i in range(len(algorithms)):
                ax.plot(n_gram, accuracy[i], color=color[i], marker='o', label=algorithms[i])

        ax.set_title('Compare Accuracy Score of N-grams between Classcifiers', fontsize=14)
        ax.set_xlabel('N-grams', fontsize=14)
        ax.set_ylabel('Accuracy Score', fontsize=14)
        ax.legend(loc='best')
        ax.set_yticks(np.arange(0, 1.02, step=0.05))

#--------------------------------------------- Precision ---------------------------------------------#
def precision_fig():
        fig, ax = plt.subplots()
        for i in range(len(algorithms)):
                ax.plot(n_gram, precision[i], color=color[i], marker='o', label=algorithms[i])

        ax.set_title('Compare Precision Score of N-grams between Classcifiers', fontsize=14)
        ax.set_xlabel('N-grams', fontsize=14)
        ax.set_ylabel('Precision Score', fontsize=14)
        ax.legend(loc='best')
        ax.set_yticks(np.arange(0, 1.02, step=0.05))

#--------------------------------------------- Recall ---------------------------------------------#
def recall_fig():
        fig, ax = plt.subplots()
        for i in range(len(algorithms)):
                ax.plot(n_gram, recall[i], color=color[i], marker='o', label=algorithms[i])

        ax.set_title('Compare Recall Score of N-grams between Classcifiers', fontsize=14)
        ax.set_xlabel('N-grams', fontsize=14)
        ax.set_ylabel('Recall Score', fontsize=14)
        ax.legend(loc='best')
        ax.set_yticks(np.arange(0, 1.02, step=0.05))

#--------------------------------------------- F1 ---------------------------------------------#
def f1_fig():
        fig, ax = plt.subplots()
        for i in range(len(algorithms)):
                ax.plot(n_gram, f1[i], color=color[i], marker='o', label=algorithms[i])

        ax.set_title('Compare F1 Score of N-grams between Classcifiers', fontsize=14)
        ax.set_xlabel('N-grams', fontsize=14)
        ax.set_ylabel('F1 Score', fontsize=14)
        ax.legend(loc='best')
        ax.set_yticks(np.arange(0, 1.02, step=0.05))

#---------------------------------------- Compare accuracy SMOTE and no SMOTE ---------------------------------------#
def compare_accuracy_noSMOTE():
        fig = plt.figure()
        title = [algorithms[i] for i in range(4)]
        for i in range(int(len(algorithms_noSMOTE)/2)):
                ax = fig.add_subplot(2, 2, i + 1)
                ax.plot(n_gram, accuracy_dataset_0[i], color= 'red', marker='o', label = 'Dataset 0 use SMOTE')
                ax.plot(n_gram, accuracy_noSMOTE_dataset_0[i], color= 'blue', marker='o', label = 'Dataset 0 no SMOTE')

                ax.plot(n_gram, accuracy_dataset_1[i], color= 'green', marker='o', label = 'Dataset 1 use SMOTE')
                ax.plot(n_gram, accuracy_noSMOTE_dataset_1[i], color= 'purple', marker='o', label = 'Dataset 1 no SMOTE')

                ax.set_title(title[i])
                ax.set_yticks(np.arange(0, 1.02, step=0.05))

        Line, Label = ax.get_legend_handles_labels()
        fig.legend(Line, labels=Label, loc="upper left", bbox_to_anchor=(0.86, 0.95))

        
#---------------------------------------- Compare precision SMOTE and no SMOTE ---------------------------------------#
def compare_precision_noSMOTE():
        fig = plt.figure()
        title = [algorithms[i] for i in range(4)]
        for i in range(int(len(algorithms_noSMOTE)/2)):
                ax = fig.add_subplot(2, 2, i + 1)
                ax.plot(n_gram, precision_dataset_0[i], color= 'red', marker='o', label = 'Dataset 0 use SMOTE')
                ax.plot(n_gram, precision_noSMOTE_dataset_0[i], color= 'blue', marker='o', label = 'Dataset 0 no SMOTE')

                ax.plot(n_gram, precision_dataset_1[i], color= 'green', marker='o', label = 'Dataset 1 use SMOTE')
                ax.plot(n_gram, precision_noSMOTE_dataset_1[i], color= 'purple', marker='o', label = 'Dataset 1 no SMOTE')

                ax.set_title(title[i])
                ax.set_yticks(np.arange(0, 1.02, step=0.05))

        Line, Label = ax.get_legend_handles_labels()
        fig.legend(Line, labels=Label, loc="upper left", bbox_to_anchor=(0.86, 0.95))
        
        
#---------------------------------------- Compare recall SMOTE and no SMOTE ---------------------------------------#
def compare_recall_noSMOTE():
        fig = plt.figure()
        title = [algorithms[i] for i in range(4)]
        for i in range(int(len(algorithms_noSMOTE)/2)):
                ax = fig.add_subplot(2, 2, i + 1)
                ax.plot(n_gram, recall_dataset_0[i], color= 'red', marker='o', label = 'Dataset 0 use SMOTE')
                ax.plot(n_gram, recall_noSMOTE_dataset_0[i], color= 'blue', marker='o', label = 'Dataset 0 no SMOTE')

                ax.plot(n_gram, recall_dataset_1[i], color= 'green', marker='o', label = 'Dataset 1 use SMOTE')
                ax.plot(n_gram, recall_noSMOTE_dataset_1[i], color= 'purple', marker='o', label = 'Dataset 1 no SMOTE')

                ax.set_title(title[i])
                ax.set_yticks(np.arange(0, 1.02, step=0.05))

        Line, Label = ax.get_legend_handles_labels()
        fig.legend(Line, labels=Label, loc="upper left", bbox_to_anchor=(0.86, 0.95))

        
        
#---------------------------------------- Compare f1 SMOTE and no SMOTE ---------------------------------------#
def compare_f1_noSMOTE():
        fig = plt.figure()
        title = [algorithms[i] for i in range(4)]
        for i in range(int(len(algorithms_noSMOTE)/2)):
                ax = fig.add_subplot(2, 2, i + 1)
                ax.plot(n_gram, f1_dataset_0[i], color= 'red', marker='o', label = 'Dataset 0 use SMOTE')
                ax.plot(n_gram, f1_noSMOTE_dataset_0[i], color= 'blue', marker='o', label = 'Dataset 0 no SMOTE')

                ax.plot(n_gram, f1_dataset_1[i], color= 'green', marker='o', label = 'Dataset 1 use SMOTE')
                ax.plot(n_gram, f1_noSMOTE_dataset_1[i], color= 'purple', marker='o', label = 'Dataset 1 no SMOTE')

                ax.set_title(title[i])
                ax.set_yticks(np.arange(0, 1.02, step=0.05))

        Line, Label = ax.get_legend_handles_labels()
        fig.legend(Line, labels=Label, loc="upper left", bbox_to_anchor=(0.86, 0.95))

vuln_statistic_fig()
compare_SMOTE_fig()
time_fig()
time_noSMOTE_fig()
accuracy_fig()
precision_fig()
recall_fig()
f1_fig()
compare_accuracy_noSMOTE()
compare_precision_noSMOTE()
compare_recall_noSMOTE()
compare_f1_noSMOTE()
plt.show()

