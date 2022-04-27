import cv2
import numpy as np
import matplotlib.pyplot as plt


#Duane Myklejord; mykle034@umn.edu; 4831000; CSCI5561, Spring 2021

#Note: Sometimes I have to run it twice to get it to work. The first time it doesn't
#recognize one of the fuctions, but if I run it again it works just fine. No idea why :/


def get_differential_filter():
    # To do

    filter_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # To do

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


def get_gradient(im_dx, im_dy):

    grad_mag = np.sqrt(np.square(im_dx) + np.square(im_dy))
    grad_angle = np.arctan(np.divide(im_dy, np.add(im_dx, 0.0001))) #0.0001 is to avoid divide by zero NaN


    #Add 180 deg. to negative values
    for i in range(np.shape(grad_angle)[0]):
        for j in range(np.shape(grad_angle)[1]):
            if grad_angle[i,j] < 0:
                grad_angle[i,j] = grad_angle[i,j]+np.pi

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size=8):

    bins = 6
    ori_histo = np.zeros(((np.shape(grad_mag)[0]//cell_size), (np.shape(grad_mag)[1]//cell_size), bins))


    for i in range(np.shape(ori_histo)[0]*cell_size):
        for j in range(np.shape(ori_histo)[1]*cell_size):
            for k in range(bins):
                if ((grad_angle[i,j] >= np.pi/bins*k-(np.pi/12)) \
                    and (grad_angle[i,j] < np.pi/bins*(k+1)-(np.pi/12)) \
                    and k>0):
                    ori_histo[(i//cell_size), (j//cell_size), k] += grad_mag[i,j]

                #if k = 0 (i.e. the split angle bin)
                elif ((grad_angle[i,j] >= 0 \
                    and grad_angle[i,j] < np.pi/bins*(1)-(np.pi/12) \
                    and k==0) \

                    or \

                    (grad_angle[i,j] >= np.pi/bins*(6)-(np.pi/12) \
                    and grad_angle[i,j] < np.pi/bins*(7)-(np.pi/6) \
                    and k==0)):

                    ori_histo[(i//cell_size), (j//cell_size), 0] += grad_mag[i,j]


    return ori_histo

def get_block_descriptor(ori_histo, block_size=2):

    ori_histo_normalized = np.zeros((np.shape(ori_histo)[0]-(block_size-1), np.shape(ori_histo)[1]-(block_size-1), 6*block_size*block_size))

    for i in range(np.shape(ori_histo_normalized)[0]):
        for j in range(np.shape(ori_histo_normalized)[1]):
            h_i = np.concatenate((ori_histo[i,j,:], ori_histo[i+1,j,:], ori_histo[i,j+1,:], ori_histo[i+1, j+1, :]))
            denominator = np.sqrt(np.sum(np.square(h_i))+0.0001)
            ori_histo_normalized[i,j,:] = np.divide(h_i, denominator)

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do

    [filter_x, filter_y] = get_differential_filter()
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)


    # # #Visualize the gradients
    # plt.imshow(im_dy, cmap='hsv')
    # plt.imshow(im_dx, cmap='hsv')
    # plt.show()

    [grad_mag, grad_angle] = get_gradient(im_dx, im_dy)

    ori_histo = build_histogram(grad_mag, grad_angle, cell_size=8) #Num cells = 8
    ori_histo_normalized = get_block_descriptor(ori_histo, block_size=2) #Block size = 2

    hog = ori_histo_normalized

    # visualize to verify
    visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

def face_recognition(I_target, I_template):

    target_hog  = extract_hog(I_target)
    template_hog = extract_hog(I_template)
    box_size = np.shape(I_template)[0]
    window_y = np.shape(template_hog)[0]
    window_x = np.shape(template_hog)[1]

    bounding_boxes = np.zeros((1,3))
    for i in range(np.shape(target_hog)[0]-window_y+1):
        for j in range(np.shape(target_hog)[1]-window_x+1):
            ncc = np.dot(template_hog[:,:,:].flatten(), target_hog[i:(i+window_y),j:(j+window_x),:].flatten())/(np.linalg.norm(template_hog[:,:,:])*np.linalg.norm(target_hog[i:(i+window_y),j:(j+window_x),:]))
            if ncc>0.6: bounding_boxes = np.append(bounding_boxes, [[j*8,i*8,ncc]], axis=0)
    bounding_boxes = np.delete(bounding_boxes, 0, 0)

    #non-maximum supression
    #find bounding box of maximum value from bounding box set
    bb = bounding_boxes.copy()
    max_bb_history = np.zeros((1,3))

    for i in range(len(bb)):

        #find local max still left in BB list (will be removed from list later)
        max_bb_index = np.where(bb[:,2] == bb[:,2].max())[0][0]

        #if the max has already gone through the supression loop, add to max_bb_history list,
        #and remove from BB list
        if bb[max_bb_index,2] in max_bb_history[:,2]:
            bb = np.delete(bb, np.where(bb[:,2] == bb[max_bb_index,2])[0][0], 0)
            max_bb_index = np.where(bb[:,2] == bb[:,2].max())[0][0] #Finds next max

        max_bb_history = np.append(max_bb_history, [bb[max_bb_index,:]], axis=0) #Adding local max to max_bb_history list

        if len(bb)==1: return np.delete(max_bb_history, 0, 0)

        max_col = int(bb[max_bb_index,1])
        max_row = int(bb[max_bb_index,0])

        #Checking for overlap
        for box in bb:
            if (box[2] not in max_bb_history[:,2]) and (box in bb): #checking all boxes except itself
                if ((int(box[1]) in range(max_col,max_col+box_size)) or (int(box[1]+box_size) in range(max_col,max_col+box_size))) and ((int(box[0]) in range(max_row,max_row+box_size)) or (int(box[0]+box_size) in range(max_row,max_row+box_size))): #I couldn't figure out how to split into multiple lines :(
                    bb_col = box[1]
                    bb_row = box[0]
                    overlapping_area = (abs(min(bb_col+box_size, max_col+box_size)) - abs(max(bb_col, max_col))) * (abs(min(bb_row+box_size, max_row+box_size)) - abs(max(bb_row, max_row)))
                    IoU = overlapping_area / np.square(box_size)
                    #Supressing any with over 50% IoU
                    if IoU > 0.5:
                        bb = np.delete(bb, np.where(bb[:,2] == box[2])[0][0], 0)


# Provided
def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()

if __name__=='__main__':


    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
