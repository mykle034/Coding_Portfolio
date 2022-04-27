import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size=32):

    class_labels, unique_index, class_labels_ind_in_arr = np.unique(label_train, return_index=True, return_inverse=True)

    # The colums are the labels in order (alphabetical), the rows are the label_train list.
    # Only one entry per row will be 1, the rest zero. The 1 will be the corresponding label.
    # (unique[column number] == corresponding label)
    label_train_hot_batch = np.zeros((np.shape(label_train)[1], len(class_labels)))
    for train_index, label_index in enumerate(class_labels_ind_in_arr):
        label_train_hot_batch[train_index, label_index] = 1

    mini_batches_indices = []
    random_indices = np.arange(0,len(label_train_hot_batch),1)
    np.random.shuffle(random_indices)
    for i in range(int(len(random_indices)/ batch_size)):
        mini_batches_indices.append(random_indices[i:i+batch_size])

    # For the partial last batch, if any:
    remainder = len(label_train_hot_batch)%batch_size
    if remainder:
        print(remainder)
        mini_batches_indices.append(random_indices[-remainder:])


    # assigning mini_batches based on the random indices
    mini_batch_x = np.empty((len(mini_batches_indices), len(im_train), batch_size))
    mini_batch_y = np.empty((len(mini_batches_indices), np.shape(label_train_hot_batch)[1] ,batch_size))
    for sub_batch_index in range(np.shape(mini_batches_indices)[0]):
        for mini_batch_index in range(np.shape(mini_batches_indices)[1]):
            mini_batch_x[sub_batch_index, :, mini_batch_index] = im_train[:,mini_batches_indices[sub_batch_index][mini_batch_index]]
            mini_batch_y[sub_batch_index, :, mini_batch_index] = label_train_hot_batch[mini_batches_indices[sub_batch_index][mini_batch_index],:]

    return mini_batch_x, mini_batch_y

def fc(x, w, b):
    y_tilde = w @ x + b

    return y_tilde


def fc_backward(dl_dy, x, w, b, y):

    dl_dx = dl_dy @ w
    dl_dw = dl_dy.T @ x.T

    dl_db = dl_dy.T

    return dl_dx, dl_dw, dl_db

def loss_euclidean(y_tilde, y):

    # y_tilde is the prediction
    # y is the ground truth label
    loss = np.sum(np.square(y_tilde - y))

    dl_dy = 2 * (y_tilde - y).T

    return loss, dl_dy

def loss_cross_entropy_softmax(x, y):

    def softmax(a): return (np.exp(a) / np.sum(np.exp(a)))

    l = -y.T @ np.log(softmax(x))

    dl_dy = (softmax(x) - y).T

    return l[0], dl_dy

def relu(x):
    epsilon = 0.01
    y = np.fmax(epsilon * x, x)

    return y


def relu_backward(dl_dy, x, y):

    dl_dx = np.where(x > 0, 1, 0.01)

    return dl_dx.T

def im2col(x,hh,ww,stride):

    """
    * Not Mine!
    Source:
    --> From website linked in assignment!
    --> https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster#im2col

    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def conv(x, w_conv, b_conv):

    x = x.reshape((14,14))
    image = np.pad(x, (1), 'constant', constant_values=0)
    image = image.reshape(1,16,16)

    image_as_col = im2col(image, 3, 3, 1)

    w_as_col = np.reshape(w_conv, (9,-1))
    b_as_col = np.reshape(b_conv, (3,-1))


    conv_as_col = image_as_col @ w_as_col + b_as_col.T

    y = conv_as_col.reshape(14,14,3)

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):

    dl_dw = np.zeros(w_conv.shape)
    dl_dx = np.zeros(x.shape)
    dl_db = np.zeros(b_conv.shape)

    x = x.reshape((14,14))
    image = np.pad(x, (1), 'constant', constant_values=0)
    image = image.reshape(1,16,16)

    image_as_col = im2col(image, 3, 3, 1)
    w_as_col = np.reshape(w_conv, (9,-1))

    dl_db = np.reshape(dl_dy, (3,-1)).T
    dl_db = np.sum(dl_db, axis=0)


    der_w_as_col = image_as_col.T @ np.reshape(dl_dy, (3,-1)).T

    dl_dw = np.reshape(der_w_as_col.T, (3,3,1,3))

    return dl_dw, np.array([dl_db])

def pool2x2(x):

    stride = 2
    size = 2
    y = np.empty((np.shape(x)[0]//2, np.shape(x)[1]//2, np.shape(x)[2]))
    max_idx = np.empty((np.shape(x)[0]//2, np.shape(x)[1]//2, np.shape(x)[2]))

    row_samples = np.arange(0, np.shape(x)[1],  stride)
    col_samples = np.arange(0, np.shape(x)[0],  stride)
    layers = np.arange(0, np.shape(x)[2], 1)
    max_locations = np.empty(1)
    for layer in layers:
        for row_idx, row in enumerate(row_samples):
            for col_idx, col in enumerate(col_samples):
                rows = np.array([row, row, row+1, row+1])
                cols = np.array([col, col+1, col, col+1])
                y[row_idx, col_idx, layer] = np.max(x[rows, cols, layer])

    return y

def pool2x2_backward(dl_dy, x, y):

    stride = 2
    size = 2
    row_samples = np.arange(0, np.shape(x)[1],  stride)
    col_samples = np.arange(0, np.shape(x)[0],  stride)
    layers = np.arange(0, np.shape(x)[2], 1)

    dl_dx = np.full_like(x, 0)
    for layer in layers:
        for row_idx, row in enumerate(row_samples):
            for col_idx, col in enumerate(col_samples):
                rows = np.array([row, row, row+1, row+1])
                cols = np.array([col, col+1, col, col+1])

                max_idx_row, max_idx_col  = np.argmax(x[rows, cols, layer].reshape((2,2)), axis=0)

                dl_dx[row+max_idx_row, col+max_idx_col, layer] = np.max(x[rows, cols, layer]) * dl_dy[row_idx,col_idx,layer]

    return dl_dx


def flattening(x):

    y = x.flatten(order='F')
    y = np.array([y]).T
    return y


def flattening_backward(dl_dy, x, y):

    dl_dx = np.reshape((dl_dy * y), np.shape(x), order='F')

    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):

    learning_rate = .1
    decay_rate = .8 # Make less than or equal to 1
    w = np.random.standard_normal((np.shape(mini_batch_y)[1], np.shape(mini_batch_x)[1]))
    b = np.random.standard_normal((10,1))
    mini_batch_loss_list = []

    for iteration in range(10000):
        if not iteration%1000:
            learning_rate = decay_rate * learning_rate

        dl_dw, dl_db, mini_batch_dw, mini_batch_db, mini_batch_loss = 0, 0, 0, 0, 0

        # The modulo division in the index keeps the mini_batches in correct range
        batch_index = iteration % (np.shape(mini_batch_x)[0])

        for image_index, image in enumerate(mini_batch_x[batch_index].T):

            image = np.array([image]).T # Making the image a 10x1

            # y_is the label
            y = np.array([mini_batch_y[batch_index, :, image_index]]).T

            # y_tilde is the prediction
            y_tilde = fc(image, w, b)

            loss, dl_dy = loss_euclidean(y_tilde, y)

            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, image, w, b, y)

            mini_batch_dw += dl_dw
            mini_batch_db += dl_db
            mini_batch_loss  += loss

        w = w - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_dw
        b = b - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_db

        if not iteration%100:
            mini_batch_loss_list.append(mini_batch_loss)

    plt.plot(mini_batch_loss_list)
    plt.show()

    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    learning_rate = 1
    decay_rate = .95 # Make less than or equal to 1

    w = np.random.standard_normal((np.shape(mini_batch_y)[1], np.shape(mini_batch_x)[1]))
    b = np.random.standard_normal((10,1))
    mini_batch_loss_list = []

    for iteration in range(10000):
        if not iteration%1000:
            learning_rate = decay_rate * learning_rate

        dl_dw, dl_db, mini_batch_dw, mini_batch_db, mini_batch_loss = 0, 0, 0, 0, 0

        batch_index = np.random.choice(np.shape(mini_batch_x)[0])

        for image_index, image in enumerate(mini_batch_x[batch_index].T):

            image = np.array([image]).T # Making the image a 10x1

            # y_is the label
            y = np.array([mini_batch_y[batch_index, :, image_index]]).T

            # y_tilde is the prediction
            y_tilde = fc(image, w, b)

            loss, dl_dy = loss_cross_entropy_softmax(y_tilde, y)

            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, image, w, b, y_tilde)

            mini_batch_dw += dl_dw
            mini_batch_db += dl_db
            mini_batch_loss  += loss


        w = w - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_dw
        b = b - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_db

        if not iteration%100:
            mini_batch_loss_list.append(mini_batch_loss/len(mini_batch_loss))

    plt.plot(mini_batch_loss_list)
    plt.show()

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):

    learning_rate = 1
    decay_rate = .8 # Make less than or equal to 1

    w1 = np.random.random_sample((30, np.shape(mini_batch_x)[1]))
    w2 = np.random.random_sample((np.shape(mini_batch_y)[1], 30))
    b1 = np.random.random_sample((30,1))
    b2 = np.random.random_sample((10,1))

    mini_batch_loss_list = []

    for iteration in range(10000):
        if not iteration%1000:
            learning_rate = decay_rate * learning_rate

        dl_dw1, dl_dw2, dl_db1, dl_db2, mini_batch_dw1, mini_batch_db1, mini_batch_dw2, mini_batch_db2, mini_batch_loss = 0, 0, 0, 0, 0, 0, 0, 0, 0

        # New way of specifiying the batch index:
        batch_index = np.random.choice(np.shape(mini_batch_x)[0])

        for image_index, image in enumerate(mini_batch_x[batch_index].T):

            image = np.array([image]).T
            # y_is the label
            y = np.array([mini_batch_y[batch_index, :, image_index]]).T

            # y_tilde is the prediction
            a1 = fc(image, w1, b1)
            f1 = relu(a1)

            a2 = fc(f1, w2, b2)

            y_tilde = a2

            loss, dl_dy = loss_cross_entropy_softmax(y_tilde, y)

            dl_da2, dl_dw2, dl_db2 = fc_backward(dl_dy, f1, w2, b2, y_tilde)

            dl_da1 = relu_backward(dl_da2, a1, f1)

            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_da1, image, w1, b1, a1)

            mini_batch_dw1 += dl_dw1
            mini_batch_db1 += dl_db1
            mini_batch_dw2 += dl_dw2
            mini_batch_db2 += dl_db2
            mini_batch_loss  += loss

        w1 = w1 - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_dw1
        b1 = b1 - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_db1
        w2 = w2 - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_dw2
        b2 = b2 - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_db2

        if not iteration%100:
            mini_batch_loss_list.append(mini_batch_loss)

    plt.plot(mini_batch_loss_list)
    plt.show()

    return w1, b1, w2, b2

def train_cnn(mini_batch_x, mini_batch_y):

    learning_rate = .1
    decay_rate = .8 # Make less than or equal to 1

    w_conv = np.random.standard_normal((3,3,1,3))
    b_conv = np.random.standard_normal((3,1))
    w_fc = np.random.standard_normal((10,147))
    b_fc = np.random.standard_normal((10,1))

    mini_batch_loss_list = []
    mini_batch_loss = 0

    for iteration in range(2000):
        if not iteration%1000:
            learning_rate = decay_rate * learning_rate
            print('loss :', mini_batch_loss, 'LR: ', learning_rate, 'iteration: ', iteration)

        dl_dw_conv, dl_dw_fc, dl_db_conv, dl_db_fc, mini_batch_dw_conv, mini_batch_db_conv, mini_batch_dw_fc, mini_batch_db_fc, mini_batch_loss = 0, 0, 0, 0, 0, 0, 0, 0, 0

        batch_index = np.random.choice(np.shape(mini_batch_x)[0])

        for image_index, image in enumerate(mini_batch_x[batch_index].T):

            y = np.array([mini_batch_y[batch_index, :, image_index]]).T

            y_convoluted = conv(image, w_conv, b_conv)

            y_relued = relu(y_convoluted)

            y_pooled = pool2x2(y_relued)

            y_flattened = flattening(y_pooled)

            y_tilde = fc(y_flattened, w_fc, b_fc)

            loss, dl_dy = loss_cross_entropy_softmax(y_tilde, y)

            dl_dx, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, y_flattened, w_fc, b_fc, y_tilde)

            unflattened_dl_dx = flattening_backward(dl_dx.T, y_pooled, y_flattened)

            un_pooled_dl_dx = pool2x2_backward(unflattened_dl_dx,  y_relued, y_pooled)

            un_relued_dl_dx = np.transpose(relu_backward(un_pooled_dl_dx, y_convoluted, y_relued)) * un_pooled_dl_dx

            mini_batch_dw_conv, mini_batch_db_conv  = conv_backward(un_relued_dl_dx, image, w_conv, b_conv, y_convoluted)

            mini_batch_dw_conv += mini_batch_dw_conv
            mini_batch_db_conv += mini_batch_db_conv
            mini_batch_dw_fc += dl_dw_fc
            mini_batch_db_fc += dl_db_fc
            mini_batch_loss += loss

        w_conv = w_conv - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_dw_conv
        b_conv = b_conv - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_db_conv.T
        w_fc = w_fc - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_dw_fc
        b_fc = b_fc - learning_rate/np.shape(mini_batch_x)[0] * mini_batch_db_fc

        if not iteration%100:
            mini_batch_loss_list.append(mini_batch_loss)
            # print('loss: ', mini_batch_loss)

    plt.plot(mini_batch_loss_list)
    plt.show()


    return w_conv, b_conv, dl_dw_fc, dl_db_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()
