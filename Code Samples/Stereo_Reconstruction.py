import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D


# Duane Myklejord
# CSCI5561: Computer Vision
# HW 5: Stereo Reconstruction
# May 6th, 2022


def find_match(img1, img2):

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)  # kp1[2].pt
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Extracting the point coordinates
    kp1_pt = cv2.KeyPoint_convert(kp1)
    kp2_pt = cv2.KeyPoint_convert(kp2)

    # For each keypoint in img2 (target), find the corresponding nearest
    # neighbors in img1 (template), returning the kp indexes in kp1
    neigh1 = NearestNeighbors(n_neighbors=2)

    neigh1.fit(des2)
    dis21, ind21 = neigh1.kneighbors(des1)      #finding the 2 closest points within des1
    # ind21 is the index of dis21 (kp2) that corresponds to the aranged indices of des1
    # == dis21 is the distances, ind21 is the index of des2 that it corresponds to

    # Ratio test
    kp21 = []
    ind21_good = []
    for i in range(len(dis21)):
        if dis21[i, 0]/dis21[i, 1] < 0.7:
            kp21.append(ind21[i, 0]) # indices of the good kp1_pt points wrt dis21
            ind21_good.append(i)     # indices of the dis21 of the good kp1_pt points

    # Now in the other direction
    neigh2 = NearestNeighbors(n_neighbors=2)
    neigh2.fit(des1)
    dis12, ind12 = neigh2.kneighbors(des2)

    # Ratio test
    kp12 = []
    ind12_good = []
    for i in range(len(dis12)):
        if dis12[i, 0]/dis12[i, 1] < 0.7:
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


    x1.astype(int)
    x2.astype(int)
    return x1, x2


def compute_F(pts1, pts2):
    # Appending a column of ones for matrix calcualtions:
    pts1 = np.hstack([pts1, np.array([np.ones(len(pts1))]).T])
    pts2 = np.hstack([pts2, np.array([np.ones(len(pts2))]).T])

    prev_inliers = 0
    best_fundamental_matrix = []
    inlier_list = []

    ransac_thr = .05
    ransac_iter = 1000
    for counter in range(ransac_iter):
        # Random sample generator, picking 8 random indices to use as correspondences
        num = range(len(pts1))
        rand_samples = np.random.choice(num, size=[8], replace=False)

        ux, uy, uz = pts1[rand_samples].T # Need the transpose because Numpy uses column vectors
        vx, vy, vz = pts2[rand_samples].T

        # Making the large A matrix for the F matrix calculation: Ax=0, x is the f_mat.
        A = np.empty((8,9))
        for i in range(len(ux)):
            A[i,:] = ([vx[i], vy[i], vz[i]] * np.array([[ux[i], uy[i], uz[i]]]).T).T.flatten()

        u, D, v = np.linalg.svd(A)
        F = np.reshape(v[-1,:], (3,3)) # Turns out it's the 8th row. I guess thats the last column of v.T

        u_tilde, D_tilde, v_tilde = np.linalg.svd(F)
        D_tilde[-1] = 0
        F_tilde = u_tilde @ np.diag(D_tilde) @ v_tilde

        F = F_tilde

        inliers = 0
        for point_idx, pts1_val in enumerate(pts1):
            # if np.abs(pts1_val @ F @ pts2[point_idx]) < ransac_thr:
            if np.abs(pts2[point_idx].T @ F @ pts1_val) < ransac_thr:
                inliers += 1

        if inliers > prev_inliers:
            best_fundamental_matrix = np.copy(F)
            prev_inliers = np.copy(inliers)

    print("max Inliers" , prev_inliers)

    return best_fundamental_matrix


def triangulation(P1, P2, pts1, pts2):
    # TO DO
    pts1 = np.hstack([pts1, np.array([np.ones(len(pts1))]).T])
    pts2 = np.hstack([pts2, np.array([np.ones(len(pts2))]).T])

    A = np.empty((len(pts1), 4, 4))
    for i in range(len(pts1)):
        A[i] = np.concatenate((pts1[i,0] * P1[2] - P1[0], # Changed the subraction's index
                                pts1[i,1] * P1[2] - P1[1],
                                pts2[i,0] * P2[2] - P2[0],
                                pts2[i,1] * P2[2] - P2[1])).reshape(4,4)

    u, D, v = np.linalg.svd(A)
    pts3D = np.empty((len(pts1), 4))
    for idx, norm_val in enumerate(v[:,-1,-1]):
        pts3D[idx] = v[idx,-1,:]/v[idx,-1,-1]

    return pts3D[:,0:3]


def disambiguate_pose(Rs, Cs, pts3Ds):
    Rs = np.array(Rs)
    Cs = np.array(Cs)

    chooser = np.zeros((4,1))
    for idx in range(4):
        for points in pts3Ds[idx]:
            if Rs[idx,:,-1] @ (points - Cs[idx].T).T > 0:
                chooser[idx] += 1
    max_pose = np.argmax(chooser)

    return Rs[max_pose], Cs[max_pose], pts3Ds[max_pose]


def compute_rectification(K, R, C):
    r_x = C / np.linalg.norm(C)
    r_z_tilde = np.array([[0,0,1]])
    r_z = r_z_tilde.T - (r_z_tilde @ r_x) * r_x / np.linalg.norm(r_z_tilde - (r_z_tilde @ r_x) * r_x)
    r_y = np.cross(r_z.T, r_x.T).T

    R_rect = np.array([r_x.T, r_y.T, r_z.T]).squeeze()
    np.shape(R_rect)

    H1 = K @ R_rect @ np.linalg.inv(K)
    H2 = K @ R_rect @ R.T @ np.linalg.inv(K)

    return H1, H2


def dense_match(img1, img2):
    pts1, pts2 = find_match(img_left_w, img_right_w)

    sift = cv2.SIFT_create()
    kp = []
    dense_feature_left = []
    dense_feature_right = []
    kptloc_l = []
    kptloc_r = []
    stride = 1
    size = 6 #8 is good, 4 is better, but 6 is best-ish
    row_samples = np.arange(size/2, np.shape(img_left_w)[0]-size/2,  stride)
    col_samples = np.arange(size/2, np.shape(img_left_w)[1]-size/2,  stride)
    for row in row_samples:
        for col in col_samples:
            kp = cv2.KeyPoint(col,row,size)
            kp_loc_l, feature_left = sift.compute(img_left_w, [kp])
            kp_loc_r, feature_right = sift.compute(img_right_w, [kp])
            dense_feature_left.append(feature_left[0])
            dense_feature_right.append(feature_right[0])
        print('completion fraction:', int(row)/max(row_samples))

    # For Debugging:
    # temp1 = np.copy(dense_feature_left)
    # temp2 = np.copy(dense_feature_right)
    # temp3 = np.copy(kptloc_l)
    # temp4 = np.copy(kptloc_r)

    dense_feature_left = np.reshape(dense_feature_left, (int(row-(size/2-1)), int(col-(size/2-1)), 128))
    dense_feature_right = np.reshape(dense_feature_right, (int(row-(size/2-1)), int(col-(size/2-1)), 128))

    disparity = np.zeros(np.shape(dense_feature_left)[0:2])
    for idx_row in range(np.shape(dense_feature_left)[0]):
        for idx_col in range(np.shape(dense_feature_left)[1]-1):
            remaining_col_idx = np.arange(idx_col, np.shape(dense_feature_left)[1]-1, 1) # Limiting the disparity search to 100 pixels
            disparity[idx_row, idx_col] = np.argmin(np.linalg.norm(dense_feature_left[idx_row, idx_col] - dense_feature_right[idx_row, remaining_col_idx], 2, axis=1))

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []

    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')

    # Added this axis ranges to keep the images the right size
    ax1.axis(xmin=0,xmax=960)
    ax2.axis(xmin=0,xmax=960)
    ax1.axis(ymin=540,ymax=0)
    ax2.axis(ymin=540,ymax=0)

    plt.show()


def find_epipolar_line_end_points(img, F, p):
    # For debugging:
    # img = img2
    # p = (x1,y1)

    # int() was addent in the p1, p2 calcualtion:
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, int(-el[2] / el[1])), (img.shape[1], int((-img_width * el[0] - el[2]) / el[1])) # int() was added to get the right type in the next line
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), np.array(p1), np.array(p2))
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # This is for verification: using cv2 functions:
    # F, inlier_bool = cv2.findFundamentalMat(pts1, pts2)
    # num_inliers = len(pts1[inlier_bool.ravel() == 1])
    # print('num inliters', num_inliers)
    # visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)


    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # i = 0
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    # exit()
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    # temp_d6 = np.copy(disparity)

    visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
