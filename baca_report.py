# -*- coding: utf-8 -*-

#%% import needed package
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

#%% additional functionality
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#%% config
random_seed = 1234
dt = DecisionTreeClassifier(max_depth=2, random_state=random_seed)

input_file = "data_mentah_baca.csv"

#%% baca data
df = pd.read_csv(input_file, sep=',')
le_kelamin = LabelEncoder()
le_bingkai = LabelEncoder()
# convert categorical column to numeric
df['kelamin'] = le_kelamin.fit_transform(df['kelamin'])
df['bingkai'] = le_bingkai.fit_transform(df['bingkai'])

# ambil features ==> kolom ketiga sampe kolom (terakhir - 1)
X = df.as_matrix(columns=df.columns[2:-1])
# ambil kelasnya ==> kolom terakhir
Y = df.as_matrix(columns=[df.columns[-1]])

# normalisasi
scaler_training = StandardScaler()
X[:, 0:-1] = scaler_training.fit_transform(X[:, 0:-1])

#%%
# visualize class
pd.value_counts(df['bingkai']).plot.bar()
plt.title('Class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
df['bingkai'].value_counts()

#%%
# begin K-folding
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
fold_num = 1
best_model = []
best_H = []
best_acc = 0
all_acc = []
class_names = ['baca01', 'baca02', 'baca03', 'baca04', 'baca05', 'baca06']
for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    print("Number of data in X_train dataset: ", X_train.shape)
    print("Number of data in y_train dataset: ", y_train.shape)
    print("Number of data in X_test dataset: ", X_test.shape)
    print("Number of data in y_test dataset: ", y_test.shape)
    
    model = dt.fit(X_train, y_train.ravel())
    
    # predicting data training
    print('Predicting on train data')
    y_train_predicted = model.predict(X_train)
    if (np.isnan(y_train_predicted).any()):
        continue
    
    cm_training = confusion_matrix(y_train, y_train_predicted.round())
    plot_confusion_matrix(cm_training, class_names, title="Confusion matrix (training), fold #{}".format(fold_num))
    print(classification_report(y_train, y_train_predicted.round()))
    
    print('Predicting on test data')
    y_test_predicted = model.predict(X_test)
    
    cm_testing = confusion_matrix(y_test, y_test_predicted.round())
    plot_confusion_matrix(cm_testing, class_names, title="Confusion matrix (testing), fold #{}".format(fold_num))
    print(classification_report(y_test, y_test_predicted.round()))
    
    # predict on all data
    predictions = model.predict(X)

#    pred_labels = lb.inverse_transform(predictions)
    num_correct = np.sum(predictions.round() == Y.ravel())
    accuracy = num_correct / len(X)
    all_acc.append(accuracy)

    print("[INFO] Fold #{} accuracy: {}".format(fold_num, accuracy))

    # premature break for debugging
    # break
    if (accuracy > best_acc):
        best_model = model
        best_acc = accuracy
        
    fold_num += 1
    
#%% overall
all_acc = np.array(all_acc)

# k-fold accuracy
print("[INFO] Accuracy (k-Folding): {} +/- {}".format(np.mean(all_acc), np.std(all_acc)))
print("[INFO] evaluating network (all dataset)...")
predictions = best_model.predict(X)
print(classification_report(Y, predictions.round()))
cm_all = confusion_matrix(Y, predictions.round())
print('[INFO] Confusion Matrix (all dataset)')
print(cm_all)
plot_confusion_matrix(cm_all, class_names, title="Confusion matrix (all dataset)")