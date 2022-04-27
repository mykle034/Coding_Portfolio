import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath

# For code timing:
import time


# run this in terminal to add env to iphython for Atom editor: python -m ipykernel install --user --name env_name

def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift_grid(img):
    cell_size = 16
    # sift = cv2.SIFT() #This line cost me 5 hours. It'll crash the ipython.
    sift = cv2.SIFT_create()
    kp = cv2.KeyPoint(7, 7, 16)

    dense_feature = []
    img_grid = np.zeros(((np.shape(img)[0]//cell_size), (np.shape(img)[1]//cell_size),128))
    for row in range(np.shape(img_grid)[0]):
        for col in range(np.shape(img_grid)[1]):
            _, feature = sift.compute(img[row:(row+cell_size), col:(col+cell_size)], [kp]) #kp must be a list, even if len==1. Spent a whole day on this. Documentation didn't mention it.
            dense_feature.append(feature[0])

    return dense_feature

def compute_dsift_sliding_window(img, stride=16, size=16):

    sift = cv2.SIFT_create()
    kp = []
    dense_feature = []
    row_samples = np.arange(size/2, np.shape(img)[1]-size/2,  stride)
    col_samples = np.arange(size/2, np.shape(img)[0]-size/2,  stride)
    for row in row_samples:
        for col in col_samples:
            kp = cv2.KeyPoint(col,row,size)
            _, feature = sift.compute(img, [kp])
            dense_feature.append(feature[0])

    return dense_feature


def get_tiny_image(img, output_size):

    img_resized = cv2.resize(img,output_size)
    img_resized = img_resized - np.mean(img_resized)
    img_resized = img_resized / np.linalg.norm(img_resized)

    feature = img_resized

    return feature


def predict_knn(feature_train, label_train, feature_test, k):

    neigh = NearestNeighbors(n_neighbors=k).fit(feature_train)
    label_test_pred_index = neigh.kneighbors(feature_test, return_distance=False)

    label_test_pred = []
    votes = []
    for x in label_test_pred_index:
        for y in x:
            votes.append(label_train[y])

        unique, position, counts = np.unique(votes,return_inverse=True, return_counts=True)
        max_index = np.bincount(position).argmax()
        label_test_pred.append(unique[max_index])
        votes = []

    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # Desired size of tiny image:
    output_size = (16,16)

    # Creating tiny images:
    feature_train = [get_tiny_image(x, output_size).flatten() for x in [cv2.imread(y, 0) for y in img_train_list]]
    feature_test = [get_tiny_image(x, output_size).flatten() for x in [cv2.imread(y, 0) for y in img_test_list]]

    # How many neighbors you'd like to vote from when determining the cluster assignment.
    k = 5
    label_pred_list = predict_knn(feature_train, label_train_list, feature_test, k)

    # Making confusion matrix
    confusion = np.zeros((len(label_classes),len(label_classes))).astype(int)
    for index, true_value in enumerate(label_test_list):
        pred_value = label_pred_list[index]
        true_class_index = label_classes.index(true_value)
        pred_class_index = label_classes.index(pred_value)
        confusion[true_class_index, pred_class_index] += 1

    #Getting percentages confusion matrix wrt percentages:
    row_sum_denominator = 1/np.tile(np.sum(confusion, axis=1), (15,1))
    confusion = np.multiply(row_sum_denominator, confusion)
    accuracy = np.mean(np.diagonal(confusion))

    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):

    # Assigning all the dense features to dic_size-number of clusters:
    kmeans = KMeans(n_clusters=dic_size).fit(dense_feature_list)
    vocab = kmeans.cluster_centers_

    return vocab


def compute_bow(feature, vocab):
    # Creates the normalized histogram for the image
    # feature is a set of SIFT features for one image, and vocab is visual dictionary.

    bin_list = predict_knn(vocab, np.arange(len(vocab))+1, feature, k=5)

    # Density=True is the normalization of the histogram
    bow_feature = np.histogram(bin_list, bins=len(vocab), range=(0,len(vocab)), density=True)

    #This can be used if vocab is passed as a kmeans type:
    # bow_feature = np.histogram(vocab.predict(feature), bins=vocab.n_clusters, range=(0,vocab.n_clusters), density=True)

    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    '''
    # This code block is only used if you want to generate the Vocab and BoW.
    # Those files are included in the folder, and this code was used to generate them.

    dic_size = 100

    # Find dense features off all images
    dense_features = np.empty((1,128)).astype(float)
    for index, img in enumerate(img_train_list):
         dense_features = np.concatenate((dense_features, compute_dsift_sliding_window(cv2.imread(img,0))))

    # Group dense features of all images into clusters, return cluster centers
    vocab  = build_visual_dictionary(dense_features, dic_size)

    # make the histogram, of length dictionary=100 and counts the occurances in each image
    bow_feature_train_list = np.array([compute_bow(compute_dsift_sliding_window(y), vocab) for y in [cv2.imread(x, 0) for x in img_train_list]], dtype=object)
    bow_feature_test_list = np.array([compute_bow(compute_dsift_sliding_window(y), vocab) for y in [cv2.imread(x, 0) for x in img_test_list]], dtype=object)

    # Convert back to list because list type is needed later.
    bow_feature_train_list = bow_feature_train_list[:,0].tolist()
    bow_feature_test_list = bow_feature_test_list[:,0].tolist()
    '''

    # To same time from running the file over and over
    # np.savetxt('vocab.out', vocab, delimiter=',')
    # np.savetxt('feature_train.out', bow_feature_train_list, delimiter=',')
    # np.savetxt('feature_test.out', bow_feature_test_list, delimiter=',')

    # Load an existing Vocab and BoW:
    vocab = np.loadtxt('vocab.out', delimiter=',')
    bow_feature_train_list = np.loadtxt('feature_train.out', delimiter=',')
    bow_feature_test_list = np.loadtxt('feature_test.out', delimiter=',')

    # Predict
    label_pred_list = predict_knn(bow_feature_train_list, label_train_list, bow_feature_test_list, k=5)

    # Making confusion matrix
    confusion = np.zeros((len(label_classes),len(label_classes))).astype(int)
    for index, true_value in enumerate(label_test_list):
        pred_value = label_pred_list[index]
        true_class_index = label_classes.index(true_value)
        pred_class_index = label_classes.index(pred_value)
        confusion[true_class_index, pred_class_index] += 1

    #Getting percentages confusion matrix wrt percentages:
    row_sum_denominator = 1/np.tile(np.sum(confusion, axis=1), (15,1))
    confusion = np.multiply(row_sum_denominator, confusion)
    accuracy = np.mean(np.diagonal(confusion))

    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test): #removed n_classes from definition
    # Make Training Datasets, one for each class.

    class_labels, unique_index, class_labels_ind_in_arr = np.unique(label_train, return_index=True, return_inverse=True)

    # The colums are the labels in order (alphabetical), the rows are the label_train list.
    # Only one entry per row will be 1, the rest zero. The 1 will be the corresponding label.
    # (unique[column number] == corresponding label)
    label_svm = np.zeros((len(label_train), len(class_labels)))
    for train_index, label_index in enumerate(class_labels_ind_in_arr):
        label_svm[train_index, label_index] = 1

    prediction_matrix = np.empty((len(feature_test), len(class_labels)))
    for svm_index, test_label in enumerate(class_labels):

        # Fit model for test_label; loops through all the labels
        svm_clf = LinearSVC(C=20).fit(feature_train, label_svm[:,svm_index])

        # Fill confidence score for each label (each label is a column)
        prediction_matrix[:,svm_index] = svm_clf.decision_function(feature_test)

    # Take the maximum confidence score in each row, and return the value (as assign to the row in label_test_pred)
    label_test_pred = class_labels[np.argmax(prediction_matrix, axis=1)]

    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    # These files were calculated from the classify_knn_bow()
    vocab = np.loadtxt('vocab.out', delimiter=',')
    bow_feature_train_list = np.loadtxt('feature_train.out', delimiter=',')
    bow_feature_test_list = np.loadtxt('feature_test.out', delimiter=',')

    # This code block is in case the pre-saved text doesn't work or exist:
    # dic_size = 100 # mm...
    #
    # # Find dense features off all images
    # dense_features = np.empty((1,128)).astype(float)
    # for index, img in enumerate(img_train_list):
    #      dense_features = np.concatenate((dense_features, compute_dsift_sliding_window(cv2.imread(img,0))))
    #
    # # Group dense features of all images into clusters, return cluster centers
    # vocab  = build_visual_dictionary(dense_features, dic_size)
    #
    # # How big is a dsift object? --all images is 400,000~ish
    #
    # # make the histogram, of length dictionary=100 and counts the occurances in each image
    # bow_feature_train_list = np.array([compute_bow(compute_dsift_sliding_window(y), vocab) for y in [cv2.imread(x, 0) for x in img_train_list]], dtype=object)
    # bow_feature_test_list = np.array([compute_bow(compute_dsift_sliding_window(y), vocab) for y in [cv2.imread(x, 0) for x in img_test_list]], dtype=object)
    #
    # # Convert back to list because list type is needed later.
    # bow_feature_train_list = bow_feature_train_list[:,0].tolist()
    # bow_feature_test_list = bow_feature_test_list[:,0].tolist()

    label_pred_list = predict_svm(bow_feature_train_list, label_train_list, bow_feature_test_list)

    # Making confusion matrix
    confusion = np.zeros((len(label_classes),len(label_classes))).astype(int)
    for index, true_value in enumerate(label_test_list):
        pred_value = label_pred_list[index]
        true_class_index = label_classes.index(true_value)
        pred_class_index = label_classes.index(pred_value)
        confusion[true_class_index, pred_class_index] += 1

    #Getting percentages confusion matrix wrt percentages:
    row_sum_denominator = 1/np.tile(np.sum(confusion, axis=1), (15,1))
    confusion = np.multiply(row_sum_denominator, confusion)
    accuracy = np.mean(np.diagonal(confusion))

    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()

    # This is in case you want to save the plots to file. The time.time() so
    # that the files don't overwrite each other.
    # plt.savefig(str(time.time()) + '.png', bbox_inches='tight') # Added this, Delete Later!


if __name__ == '__main__':
    # To do: replace with your dataset path

    start = time.time()
    print("Duane Myklejord; MYKLE034; CSCI5561; HW3; MARCH 18,2022",'\n','Make sure the Vocab and BoW files are in the right folder.')
    print('If you would like to generate the Vocab and BoW, un-comment the relevant code sections.')

    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    end = time.time()
    print('Run time:',int(end - start), '[seconds]')
