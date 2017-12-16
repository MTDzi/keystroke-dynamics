import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from copy import deepcopy


def plot_user_sequences(DF_whole, user_name, sequence_processor=None):
    if sequence_processor is None:
        sequence_processor = lambda x: x
        title = 'Sequence "as-is"'
        ylabel = 'sequence'
    else:
        processor_name = sequence_processor.__name__
        title = 'Plot for "{}"'.format(processor_name)
        ylabel = processor_name + '(sequence)'

    DF_one_user_sequences = DF_whole.query(
        '(user_name == @user_name) and (registration == 1)'
    )['sequence']
    for seq in DF_one_user_sequences:
        seq = sequence_processor(seq)
        _ = plt.plot(seq, alpha=0.5, marker='o', linestyle='--')

    base_fontsize = 20
    plt.title(title, fontsize=base_fontsize)
    plt.xlabel('index', fontsize=base_fontsize-4)
    plt.ylabel(ylabel, fontsize=base_fontsize-4)
    # To ensure the xticks are integers
    tick = int(len(seq)/10) + 1
    plt.xticks(range(0, len(seq), tick))
    plt.show()


def plot_auc(y_true, y_pred):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 10));

    fontsize = 20
    plt.xlim([-0.05, 1.05])
    plt.xticks(fontsize=fontsize)
    plt.xlabel('False Positive Rate', fontsize=fontsize)

    plt.ylim([-0.05, 1.05])
    plt.yticks(fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)

    plt.title('Receiver operating characteristic', fontsize=fontsize+4)

    lw = 4
    plt.plot(fpr, tpr, color='darkviolet',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc);

    plt.legend(loc='lower right', fontsize=fontsize)
