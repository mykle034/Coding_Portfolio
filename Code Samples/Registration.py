import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

# Duane Myklejord
# CSCI5561, HW2


def find_match(img1, img2):

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)  # kp1[2].pt
    kp2, des2 = sift.detectAndCompute(img2, None)

    img1 = cv2.drawKeypoints(img1, kp1, img1)
    img2 = cv2.drawKeypoints(img2, kp2, img2)

    # Extracting the point coordinates
    kp1_pt = cv2.KeyPoint_convert(kp1)
    kp2_pt = cv2.KeyPoint_convert(kp2)

    # For each keypoint in img2 (target), find the corresponding nearest
    # neighbors in img1 (template), returning the kp indexes in kp1
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(des2)
    dis21, ind21 = neigh.kneighbors(des1)

    kp21 = []
    ind21_good = []
    for i in range(len(dis21)):
        if dis21[i, 0]/dis21[i, 1] < 0.8:
            kp21.append(ind21[i, 0])
            ind21_good.append(i)

    # Now in the other direction
    neigh.fit(des1)
    dis12, ind12 = neigh.kneighbors(des2)
    kp12 = []
    ind12_good = []
    for i in range(len(dis12)):
        if dis12[i, 0]/dis12[i, 1] < 0.8:
            kp12.append(ind12[i, 0])
            ind12_good.append(i)

    x1 = np.empty([0, 2])
    x2 = np.empty([0, 2])

    # maintaining consistancy between the two sets of points
    # kp21 index of kp2 that corresponds to
    for i in range(len(ind12_good)):
        if (ind12_good[i] in kp21):
            if ind21_good[kp21.index(ind12_good[i])] in kp12:
                kp12_index = kp12.index(ind21_good[kp21.index(ind12_good[i])])
                kp21_index = kp21.index(ind12_good[i])
                x1 = np.append(x1, [kp1_pt[kp12[kp12_index]]], axis=0)
                x2 = np.append(x2, [kp2_pt[kp21[kp21_index]]], axis=0)

    x1 = x1.astype(int)
    x2 = x2.astype(int)

    return(x1, x2)


def align_image_using_feature(x1, x2, ransac_thr=10, ransac_iter=1000):

    prev_inliers = 0
    best_transform = []

    # Append a column of ones for the matrix transforms later in the funciton
    x2 = np.append(x2, np.transpose([np.ones(len(x2))]).astype(int), axis=1)


    for counter in range(ransac_iter):
        # Random sample generator:
        x, y, z = [], [], []
        num = range(len(x1))
        while x == y or x == z or y == z:
            x = np.random.choice(num)
            y = np.random.choice(num)
            z = np.random.choice(num)
        rand_samples = [x, y, z]
        # print(x, y, z)
        [u1, v1], [u2, v2], [u3, v3] = x1[rand_samples, :]
        [u1p, v1p], [u2p, v2p], [u3p, v3p] = x2[rand_samples, 0:2]

        # Affine transform generator:
        A = np.array([[u1, v1, 1, 0, 0, 0], [0, 0, 0, u1, v1, 1],
                      [u2, v2, 1, 0, 0, 0], [0, 0, 0, u2, v2, 1],
                      [u3, v3, 1, 0, 0, 0], [0, 0, 0, u3, v3, 1]])

        # #Example checking:
        # u1, v1, u2, v2, u3, v3 = 0, 0, 1, 0, 0, 1
        # u1p, v1p, u2p, v2p, u3p, v3p = 3, -2, 4.2, -0.4, 2, 0
        # Ans == 1.2, -1, 3; 1.6, 2, -2; 0, 0, 1

        # Finding the transform (x_mat)
        b = np.array([u1p, v1p, u2p, v2p, u3p, v3p]).T
        try:
            a11, a12, a13, a21, a22, a23 = np.linalg.inv(A.T @ A) @ A.T @ b
        except np.linalg.LinAlgError:
            temp = 0
        x = np.array([a11, a12, a13, a21, a22, a23])
        x_mat = np.array([[a11, a12, a13], [a21, a22, a23], [0, 0, 1]])

        # Calculating the x2 points through the transform
        # x1 * A = x2
        x2_calc = np.empty([0, 3])
        for i in range(len(x1)):
            x2_calc = np.append(x2_calc, [x_mat @ np.append(x1[i], 1)], axis=0)
        x2_calc = x2_calc.astype(int)

        # Finding the distance between the actual and calculated points
        distance = np.empty([0, 1])
        for i in range(len(x2)):
            distance = np.append(distance, np.linalg.norm(x2[i]-x2_calc[i]))

        # Counting the inliers (points) that are within the threshold
        inliers = 0
        for x in distance:
            if x < ransac_thr:
                inliers += 1
        if inliers > prev_inliers:
            best_transform = np.copy(x_mat)
            prev_inliers = np.copy(inliers)

            # # Visualizations for debugging:
            # viz1_mat = np.array([[u1, v1], [u2, v2], [u3, v3]])
            # viz2_mat = np.array([[u1p, v1p], [u2p, v2p], [u3p, v3p]])
            # visualize_find_match(template, target_list[0], viz1_mat, viz2_mat)

    return best_transform


def warp_image(img, A, output_size):

    # Trying to do the interpolation using vectors:
    x = np.arange(0,output_size[1]) #Columns
    y = np.arange(0,output_size[0]) #Rows
    xx, yy = np.meshgrid(x,y)
    img_2 = np.array([xx.flatten(), yy.flatten(), np.ones(len(x)*len(y))])

    # The corresponding inverse-warped coordinates
    img_invwarp = (np.transpose(img_2) @ np.transpose(A))

    img_warped = interpolate.interpn((np.arange(np.shape(img)[1]), np.arange(np.shape(img)[0])), np.transpose(img), (img_invwarp[:,0:2]), bounds_error=False)

    img_warped  = img_warped.reshape(output_size)

    return img_warped


def get_differential_filter():

    filter_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    return filter_x, filter_y


def filter_image(im, filter):

    #Pad outer pixel of original image (top and bottom)
    num_col = np.shape(im)[1]
    num_row = np.shape(im)[0]
    padding = [np.zeros(num_col)]
    im = np.append(padding, im)
    im = np.append(im, padding)
    im.shape = (num_row + 2, num_col)

    #Flipping the matrix and re-doing the operation:
    im = im.T
    num_col = np.shape(im)[1]
    num_row = np.shape(im)[0]
    padding = [np.zeros(num_col)]
    im = np.append(padding, im)
    im = np.append(im, padding)
    im.shape = (num_row + 2, num_col)
    im = im.T

    #Filtering the image:
    im_filtered = np.zeros((np.shape(im)[0]-2, np.shape(im)[1]-2))

    for i in range(im_filtered.shape[0]):
        for j in range(im_filtered.shape[1]):
            im_filtered[i,j] = np.dot(filter.reshape(-1), (im[i:(i+3), j:(j+3)]).reshape(-1))

    scaling = float(np.max(im_filtered))
    im_filtered  = [x/scaling for x in im_filtered]

    return im_filtered


def align_image(template, target, A):

    x = np.arange(0,np.shape(template)[1]) #Columns
    y = np.arange(0,np.shape(template)[0]) #Rows

    # Get gradient
    [filter_x, filter_y] = get_differential_filter()
    im_dx = np.array(filter_image(template, filter_x))
    im_dy = np.array(filter_image(template, filter_y))
    im_grad  = np.array([im_dx.flatten(), im_dy.flatten()])

    # Compute the Jacobian
    jacobian = np.empty([2,6, len(x)*len(y)])
    index = 0
    for u in x:
        for v in y:
            jacobian[:,:, index] = np.array([[u, v, 1, 0, 0, 0], [0, 0, 0, u, v, 1]])
            index += 1

    # Compute the steepest descent images
    sd_img = np.empty([len(im_grad[0]),6])
    index = 0
    for u in x:
        for v in y:
            sd_img[index,:] = np.transpose(im_grad[:,index]) @ jacobian[:,:,index]
            index += 1

    # Plotting the steepest descent images
    sd_img_plotting = np.transpose(sd_img).reshape([6, np.shape(template)[0], np.shape(template)[1]])
    fig, axeslist = plt.subplots(ncols=6, nrows=1)
    for ind,title in enumerate(sd_img_plotting[:]):
        axeslist.ravel()[ind].imshow(sd_img_plotting[ind], cmap=plt.gray())
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.show()

    # Computing the hessian
    hessian = np.transpose(sd_img) @ (sd_img)

    # Inverse Compositional Iterations
    errors = []

    #Using a for loop to elminate infinite looping in case of non-convergence
    for i in range(100):
        error_img = warp_image(target, A, np.shape(template)) - template
        if (error_img == error_img).all(): #Keeping the last valid error matrix
            errors.append(np.linalg.norm(error_img)) #Keeping track of the errors

        F = np.transpose(sd_img) @ error_img.flatten()
        delta_p = np.linalg.inv(hessian) @ F

        # Shorthand: inv_delta_p = inverse(delta_p + [[1,0,0],[0,1,0],[0,0,0]])
        inv_delta_p = np.linalg.inv(np.append(delta_p.reshape(2,3), [0,0,1]).reshape(3,3) - np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]))
        A = (A - np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])) @ inv_delta_p

        # Keeping the last valid A matrix:
        if (A == A).all():
            A_refined = A

        # print("norm of delta p: ", np.linalg.norm(delta_p))
        if np.linalg.norm(delta_p) < 30:
            # print("norm(delta_p) is less than 30")
            break

    print("the iterations did not converge")

    return  A_refined, errors


def track_multi_frames(template, img_list):

    # This is how I would write the code for this function.
    # Unfortunatly, because I couldn't figure out how to
    # properly update my tranform, this doesn't work.
    # Thus, I just used the approx. A from feature matching
    # to get a rough tracking vizualization.

    # A_list = []
    # x1, x2 = find_match(template, img)
    # A = align_image_using_feature(x1, x2)
    # for img in img_list:
    #     template = warp_image(img, A, template.shape)
    #     A_refined, errors = align_image(template, target_list[0], A)
    #     A_list.append(A_refined)
    # return A_list

    # This just uses the approx. transform from the feature matching.
    A_list = []
    for img in img_list:
        x1, x2 = find_match(template, img)
        A = align_image_using_feature(x1, x2)
        A_list.append(A)

    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()


def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                          [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg',
                          0)  # read as grey scale image
    target_list = []
    for i in range(4):
        # read as grey scale image
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 10
    ransac_iter = 1000
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)
