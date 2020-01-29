import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import seaborn as sns


def show_image(image):
    plt.imshow(image)
    plt.imshow(image)


def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Blues',
                          figsize=(14, 7), fontsize=14):
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + np.finfo(np.float32).eps)

    confusion_matrix = confusion_matrix.astype(np.int)
    # df_cm = pd.DataFrame(
    #     confusion_matrix, index=classes, columns=classes,
    # )
    df_cm = pd.DataFrame(confusion_matrix)
    # fig = plt.figure(figsize=figsize)
    fig = plt.figure()
    try:
        # heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='plasma',
        #                       linecolor='black', linewidths=1)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='hot')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    # heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    # heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
