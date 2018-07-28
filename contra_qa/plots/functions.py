import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix  # noqa
import itertools  # noqa


def simple_step_plot(ylist,
                     yname,
                     title,
                     path,
                     figsize=(4, 4),
                     labels=None):
    """
    Plots values over time.

    :param ylist: list of values lists
    :type ylist: list
    :param yname: value name
    :type yname: str
    :param title: plot's title
    :type title: str
    :param path: path to save plot, use None for not saving
    :type path: str
    :param figsize: plot's size
    :type figsize: tuple
    :param labels: label for each values list in ylist
    :type range_points: list
    """
    y0 = ylist[0]
    x = np.arange(1, len(y0) + 1, 1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for y in ylist:
        ax.plot(x, y)
    plt.xlabel('step')
    plt.ylabel(yname)
    plt.title(title,
              fontsize=14,
              fontweight='bold')
    plt.grid(True)
    if labels is not None:
        plt.legend(labels,
                   loc='upper left')
    if path is not None:
        plt.savefig(path)


def bar_plot(Xaxis,
             Yaxis,
             df,
             title,
             path,
             xlabel="task",
             ylabel="accuracy",
             hue=None,
             figsize=(9, 9)):
    """
    Plot a barplot

    :param Xaxis: collum
    :type Xaxis: str
    :param Yaxis: collum
    :type Yaxis: str
    :param df: data
    :type df: pd.DataFrame
    :param title: plot's title
    :type title: str
    :param path: path to save plot, use None for not saving
    :type path: str
    :param figsize: plot's size
    :type figsize: tuple
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = sns.barplot(x=Xaxis, y=Yaxis, hue=hue, data=df)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(), decimals=6),
                    (p.get_x() + p.get_width() / 2.,
                     p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontweight='bold',
                    color='black')
    fig.suptitle(title, fontsize=18, fontweight='bold')
    if path is not None:
        plt.savefig(path)


def plot_histogram_from_labels(labels, labels_legend, path):
    """
    Plot dataset histogram
    :param label_path: array of labels
    :type label_path: np.array
    :param labels_legend: list with the name of labels
    :type labels_legend: list
    :param path: name to save histogram
    :type path: np.str
    """

    data_hist = plt.hist(labels,
                         bins=np.arange(len(labels_legend) + 1) - 0.5,
                         edgecolor='black')
    axes = plt.gca()
    axes.set_ylim([0, len(labels)])

    plt.title("Histogram of {} data points".format(len(labels)))
    plt.xticks(np.arange(len(labels_legend) + 1), labels_legend)
    plt.xlabel("Label")
    plt.ylabel("Frequency")

    for i in range(len(labels_legend)):
        plt.text(data_hist[1][i] + 0.25,
                 data_hist[0][i] + (data_hist[0][i] * 0.01),
                 str(int(data_hist[0][i])))
    plt.savefig(path)
    plt.show()
    plt.close()


def plot_confusion_matrix(cm,
                          classes,
                          title,
                          normalize=False,
                          cmap=plt.cm.Oranges,
                          path="confusion_matrix.png"):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    'cmap' controls the color plot. colors:
    https://matplotlib.org/1.3.1/examples/color/colormaps_reference.html
    :param cm: confusion matrix
    :type cm: np array
    :param classes: number of classes
    :type classes: int
    :param title: image title
    :type title: str
    :param cmap: plt color map
    :type cmap: plt.cm
    :param path: path to save image
    :type path: str
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10.3, 7.8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=24, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes, rotation=45)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(path)


def plotconfusion(truth, predictions, path, label_dict, classes):
    """
    This function plots the confusion matrix and
    also prints useful statistics.
    :param truth: true labels
    :type truth: np array
    :param predictions: model predictions
    :type predictions: np array
    :param path: path to save image
    :type path: str
    :param label_dict: dict to transform int to str
    :type label_dict: dict
    :param classes: list of classes
    :type classes: list
    """
    acc = np.array(truth) == np.array(predictions)
    size = float(acc.shape[0])
    acc = np.sum(acc.astype("int32")) / size
    truth = [label_dict[i] for i in truth]
    predictions = [label_dict[i] for i in predictions]
    cm = ConfusionMatrix(truth, predictions)
    print(cm)
    cm_array = cm.to_array()
    cm_diag = np.diag(cm_array)
    sizes_per_cat = []
    for n in range(cm_array.shape[0]):
        sizes_per_cat.append(np.sum(cm_array[n]))
    sizes_per_cat = np.array(sizes_per_cat)
    sizes_per_cat = sizes_per_cat.astype(np.float32) ** -1
    recall = np.multiply(cm_diag, sizes_per_cat)
    print("\nRecall:{}".format(recall))
    print("\nRecall stats: mean = {0:.6f}, std = {1:.6f}\n".format(np.mean(recall),  # noqa
                                                                    np.std(recall)))  # noqa
    title = "Confusion matrix of {0} examples\n accuracy = {1:.2f}".format(int(size),  # noqa
                                                                           acc)
    plot_confusion_matrix(cm_array, classes, title=title, path=path)
    cm.print_stats()