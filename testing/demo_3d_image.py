import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
import re
from sklearn.cluster import KMeans

import glob, os


# reflectors
reflectors =  [
        "NONE",
        "F_SPINEMID",
        "F_R_CHEST",
        "F_L_CHEST",
        "F_R_HEAD",
        "F_L_HEAD",
        "B_HEAD",
        "B_BACK",
        "B_SPINEMID",
        "B_R_SHOULDER",
        "F_R_SHOULDER",
        "R_ELBOW",
        "R_WRIST",
        "R_HAND",
        "B_L_SHOULDER",
        "F_L_SHOULDER",
        "L_ELBOW",
        "L_WRIST",
        "L_HAND",
        "R_PELVIS",
        "R_CALF",
        "R_ANKLE",
        "R_FOOT",
        "L_PELVIS",
        "L_CALF",
        "L_ANKLE",
        "L_FOOT",
        "NOSE"]

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[1, 2], [1, 3], [2, 10], [10, 11], [11, 12], [12, 13], [3, 15], [15, 16], [16, 17], \
[17, 18], [1, 19], [19, 20], [20, 21], [21, 22], [1, 23], [23, 24], [24, 25], [25, 26], \
[4, 6], [5, 6], [6, 7], [7, 8], [8, 19], [8, 23], [7, 9], [9, 11], [7, 14], [14, 16]]

# the middle joints heatmap correpondence
mapIdx = [[28,29], [29,30], [31,32], [33,34], [35,36], [37,38], [39,40], [41,42], \
          [43,44], [45,46], [47,48], [49,50], [51,52], [53,54], [55,56], \
          [57,58], [59,60], [61,62], [63,64], [65,66], [67,68], [69, 70], 
          [71, 72], [73, 74], [75, 76], [77, 78], [79, 80]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [17, 145, 170], [89, 145, 240],
          [200, 55, 163], [45, 20, 240], [110, 255, 23], [176, 0, 12], [105, 100, 70], [70, 70, 70],
          [25, 96, 189]]


def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return 0 # or (0,0,0,0) ?
  return (x, y, w, h)

def isMergedRegion(a,b):
  inter = intersection(a, b)
  if inter != 0:
    e_in = inter[2]*inter[3]
    e_a = a[2]*a[3]
    e_b = b[2]*b[3]

    print(str(e_in) + " " + str(e_a) + " " + str(e_b))
    if (np.abs(e_in - e_a) < 4) and (np.abs(e_in - e_b) < 4):
        return True
    else:
        return False

def cluster(data, maxgap):
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*

        >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

        >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

    '''
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups

def cluster2(data, maxgap):
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*

        >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

        >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

    '''
    # data.sort()
    groups = [[data[0]]]
    for x in range(len(data)):
        if abs(data[x][2] - groups[-1][-1][2]) <= maxgap:
            groups[-1].append(data[x])
        else:
            groups.append([data[x]])

    toDel = []
    for i in range(len(groups)):
        if len(groups[i]) < 15:
            toDel.append(groups[i])
    
    groups = [x for x in groups if x not in toDel]

    return groups

def process (input_image, params, model_params):   


    num_of_heatmaps = 27
    num_of_PAFs = 56
    num_of_PAFs_normal = 28

    oriImg = cv2.imread(input_image)  # B,G,R order
    rawDepth = read_pgm(input_image.replace("mc_blob.png", "depth.pgm"), byteorder='>')
    # pyplot.imshow(rawDepth, pyplot.cm.gray)
    # pyplot.show()
    # multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], num_of_heatmaps))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], num_of_PAFs))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(num_of_heatmaps):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.1)) # params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 4

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - num_of_PAFs_normal for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, 3 * score_with_dist_prior,
                                                     3 * score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, num_of_PAFs_normal + 1))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < num_of_heatmaps - 1:
                    row = -1 * np.ones(num_of_PAFs_normal + 1)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    # subset = np.delete(subset, deleteIdx, axis=0)

    canvas = cv2.imread(input_image)  # B,G,R order
   
    all_peaks_max_index = np.zeros(num_of_heatmaps - 1, dtype=int)
    for i in range(num_of_heatmaps - 1):
        if len(all_peaks[i]) > 0:
            max_value = 0
            for j in range(len(all_peaks[i])):
                if max_value < all_peaks[i][j][2]:
                    max_value = all_peaks[i][j][2]
                    all_peaks_max_index[i] = j

    deleteIdReflector = []
    for i in range(num_of_heatmaps - 1):
        if len(all_peaks[i]) > 0:
            for j in range(num_of_heatmaps - 1):
                if i != j and len(all_peaks[j]) > 0:
                    vec = np.subtract(all_peaks[i][all_peaks_max_index[i]][:2], all_peaks[j][all_peaks_max_index[j]][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm < 6:
                        if (all_peaks[i][all_peaks_max_index[i]][2] > all_peaks[j][all_peaks_max_index[j]][2]):
                            deleteIdReflector.append(j)
                        else:
                            deleteIdReflector.append(i)
    for i in range(len(deleteIdReflector)):
        all_peaks[deleteIdReflector[i]] = []

    file_3d.write(str(frameIndex) + '\n')
    file_3d.write('NONE {  }\n')
    
    detected_contour_depth_values = []
    detected_contour_coordinates = []
    detected_rectangles = []
    detected_ids = []
    merged_sets = []

    for i in range(num_of_heatmaps - 1):
        if len(all_peaks[i]) > 0 and all_peaks[i] != []:
            # cv2.circle(canvas, all_peaks[i][all_peaks_max_index[i]][0:2], 4, colors[i], thickness=4)
            # Copy the thresholded image.
            im_floodfill = canvas.copy()
            
            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = canvas.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            
            # Floodfill from point (0, 0)
            flood_return = cv2.floodFill(im_floodfill, mask, all_peaks[i][all_peaks_max_index[i]][0:2], [255,255,255])

            for j in range(len(detected_rectangles)):
                 if (detected_ids[j] != i):
                    if isMergedRegion(detected_rectangles[j], flood_return[3]):

                        # if ()
                        # del detected_rectangles[j]
                        # break
                        merged_sets.append([i, detected_ids[j]])
                        
                        # cv2.imshow("image", flood_return[1])
                        # cv2.waitKey(0)
            detected_ids.append(i)
            detected_rectangles.append(flood_return[3])

            
            # Invert floodfilled image
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            
            # Combine the two images to get the foreground.
            fill_image = canvas | im_floodfill_inv
            
            mask_gray = cv2.cvtColor(fill_image, cv2.COLOR_BGR2GRAY)
            # mask_gray = cv2.normalize(src=mask_gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)    

            im2, contours, hierarchy = cv2.findContours(mask_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 1:
                
                values = np.zeros((contours[1].shape[0]), dtype=float)
                for p in range(contours[1].shape[0]):
                    depth_value = rawDepth[contours[1][p][0][1]][contours[1][p][0][0]]
                    # print(str(i) + " " + str(depth_value))
                    if depth_value > 1000 and depth_value < 2500 and contours[1][p][0][1] > 10 and contours[1][p][0][1] < 400 and contours[1][p][0][0] > 40 and contours[1][p][0][0] < 400:
                        values[p] = depth_value
                    else:
                        values[p] = 0
                    if (values[p] > 0):
                        detected_contour_coordinates.append([contours[1][p][0][1], contours[1][p][0][0], values[p]])
                    
                detected_contour_coordinates.sort(key=lambda x : x[2]) # = np.sort(detected_contour_coordinates, axis=2)
                values[::-1].sort()
                values = [x for x in values if x > 0]
                # zeros = np.count_nonzero(values)
                # # values = np.nanmean(values[:])
                # depth = values[int((len(values) - zeros)/2)]
                if np.median(values) != np.nan and np.median(values) > 0:
                    detected_contour_depth_values.append(values)
                else:
                    del detected_ids[-1]
                    del detected_rectangles[-1]
            else:
                del detected_ids[-1]
                del detected_rectangles[-1]
                # del merged_sets[-1]

    # if (len(merged_sets) > 0):
    ## Clustering
    temp_detected = [x for x in detected_ids]

    for i in range(len(merged_sets)):
        if (merged_sets[i][0] in temp_detected) and (merged_sets[i][1] in temp_detected):
            # clusters = cluster(detected_contour_depth_values[detected_ids[i]], 65)
            # clusters = cluster2(detected_contour_coordinates, 200)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(detected_contour_coordinates)

            detected_contour_depth_values[detected_ids.index(merged_sets[i][0])] = [x[2] for x in detected_contour_coordinates if kmeans.labels_[detected_contour_coordinates.index(x)] == 0]
            detected_contour_depth_values[detected_ids.index(merged_sets[i][1])] = [x[2] for x in detected_contour_coordinates if kmeans.labels_[detected_contour_coordinates.index(x)] == 1]

            # if (len(clusters) == 2):
            #     all_xy = []
            #     all_xy.append([x for x in clusters[0][:]][0])
            #     x_mean_a = [x for x in clusters[0][:]].mean()
            #     y_mean_a = clusters[0][:][1].mean()

            #     x_mean_b = clusters[0][:][0].mean()
            #     y_mean_b = clusters[0][:][1].mean()

            #     dx_a = np.abs(all_peaks[merged_sets[i][0]][all_peaks_max_index[merged_sets[i][0]]][0] - x_mean_a)
            #     dy_a = np.abs(all_peaks[merged_sets[i][0]][all_peaks_max_index[merged_sets[i][0]]][1] - y_mean_a)

            #     dx_b = np.abs(all_peaks[merged_sets[i][0]][all_peaks_max_index[merged_sets[i][0]]][0] - x_mean_b)
            #     dy_b = np.abs(all_peaks[merged_sets[i][0]][all_peaks_max_index[merged_sets[i][0]]][1] - y_mean_b)
                
            #     da = dx_a*dx_a + dy_a*dy_a
            #     db = dx_b*dx_b + dy_b*dy_b

            #     if (da < db):
            #         detected_contour_depth_values[temp_detected.index(merged_sets[i][0])] = [x for x in clusters[0][:][2]]
            #         detected_contour_depth_values[temp_detected.index(merged_sets[i][1])] = [x for x in clusters[1][:][2]]
            #     else:
            #         detected_contour_depth_values[temp_detected.index(merged_sets[i][0])] = [x for x in clusters[1][:][2]]
            #         detected_contour_depth_values[temp_detected.index(merged_sets[i][1])] = [x for x in clusters[2][:][2]]
            
                                

            temp_detected.remove(merged_sets[i][0])
            temp_detected.remove(merged_sets[i][1])


    detected_index = 0
    for i in range(num_of_heatmaps - 1):
        if (i in detected_ids):           
            depth = np.median(detected_contour_depth_values[detected_index])
            if "D4" in input_image:
                vec3 = [KRT4_x[all_peaks[i][all_peaks_max_index[i]][0]][int(all_peaks[i][all_peaks_max_index[i]][1])]*depth, KRT4_y[all_peaks[i][all_peaks_max_index[i]][0]][int(all_peaks[i][all_peaks_max_index[i]][1])]*depth, depth, 1000.0]
                vec3 = np.true_divide(vec3, 1000.0)
                
                # trans = np.transpose(Ext4)
                # final_vec3 = np.matmul(trans, vec3, out=None)

                final_vec3 = np.matmul(Ext4, vec3, out=None)

                file_3d.write(reflectors[i+1] + ' { ' + str(final_vec3[0]) + ' ' + str(final_vec3[1]) + ' ' + str(final_vec3[2]) + ' ' + str(final_vec3[0]) + ' ' + str(final_vec3[1]) + ' ' + str(final_vec3[2]) + ' }\n')
            elif "D6" in input_image:
                vec3 = [KRT6_x[all_peaks[i][all_peaks_max_index[i]][0]][int(all_peaks[i][all_peaks_max_index[i]][1])]*depth, KRT6_y[all_peaks[i][all_peaks_max_index[i]][0]][int(all_peaks[i][all_peaks_max_index[i]][1])]*depth, depth, 1000.0]
                vec3 = np.true_divide(vec3, 1000.0)
                
                # trans = np.transpose(Ext6)
                # final_vec3 = np.matmul(trans, vec3, out=None)

                final_vec3 = np.matmul(Ext6, vec3, out=None)
                
                file_3d.write(reflectors[i+1] + ' { ' + str(final_vec3[0]) + ' ' + str(final_vec3[1]) + ' ' + str(final_vec3[2]) + ' ' + str(final_vec3[0]) + ' ' + str(final_vec3[1]) + ' ' + str(final_vec3[2]) + ' }\n')
            elif "D8" in input_image:
                vec3 = [KRT8_x[all_peaks[i][all_peaks_max_index[i]][0]][int(all_peaks[i][all_peaks_max_index[i]][1])]*depth, KRT8_y[all_peaks[i][all_peaks_max_index[i]][0]][int(all_peaks[i][all_peaks_max_index[i]][1])]*depth, depth, 1000.0]
                vec3 = np.true_divide(vec3, 1000.0)
                # trans = np.transpose(Ext8)
                # final_vec3 = np.matmul(trans, vec3, out=None)

                final_vec3 = np.matmul(Ext8, vec3, out=None)

                file_3d.write(reflectors[i+1] + ' { ' + str(final_vec3[0]) + ' ' + str(final_vec3[1]) + ' ' + str(final_vec3[2]) + ' ' + str(final_vec3[0]) + ' ' + str(final_vec3[1]) + ' ' + str(final_vec3[2]) + ' }\n')

            # cv2.drawContours(im2, contours, -1, (255,255,255), 1)

            # cv2.imwrite(input_image.replace(".png", "_new.png"), im2)
            # ret = cv2.floodFill(canvas, None, all_peaks[i][all_peaks_max_index[i]][0:2], colors[i])
            # cv2.imshow(str(1), im2)
            # cv2.waitKey(0)

            detected_index += 1
            # print("done")
        else:
            file_3d.write(reflectors[i+1] + ' {  }\n')
            # cv2.imshow(str(1), im2)
            # cv2.waitKey(0)
    else:
        file_3d.write(reflectors[i+1] + ' {  }\n')

            # cv2.putText(canvas, str(i+1), all_peaks[i][all_peaks_max_index[i]][0:2], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA) 
        # for j in range(len(all_peaks[i])):
        #     cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    file_3d.write('NOSE {  }\n')
    stickwidth = 4

    # for i in range(num_of_heatmaps):
    #     # for n in range(len(subset)):
    #     #     index = subset[n][np.array(limbSeq[i]) - 1]
    #     #     if -1 in index:
    #     #         continue
    #     for n in range(len(connection_all)):
    #         if len(connection_all[n]):
    #             partAs = connection_all[n][:, 0]
    #             partBs = connection_all[n][:, 1]
    #             indexA, indexB = np.array(limbSeq[n]) - 1

    #             cur_canvas = canvas.copy()
    #             Y = candidate[indexA, 0]
    #             X = candidate[indexB, 1]
    #             mX = np.mean(X)
    #             mY = np.mean(Y)
    #             length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
    #             angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
    #             polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
    #                                     360, 1)
    #             cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
    #             canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    
    #### PAPER FIGURE
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    ####

    for i in range(len(limbSeq)):
        if len(all_peaks[limbSeq[i][0]-1]) > 0 and len(all_peaks[limbSeq[i][1]-1]) > 0:
            cur_canvas = canvas.copy()
            Y = all_peaks[limbSeq[i][0] - 1][all_peaks_max_index[limbSeq[i][0] - 1]]
            X = all_peaks[limbSeq[i][1] - 1][all_peaks_max_index[limbSeq[i][1] - 1]]
            mX = (X[1] + Y[1]) / 2
            mY = (X[0] + Y[0]) / 2
            length = ((X[0] - Y[0]) ** 2 + (X[1] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[1] - Y[1], X[0] - Y[0]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                    360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[limbSeq[i][0] - 1])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)


    for i in range(num_of_heatmaps - 1):
        if len(all_peaks[i]) > 0:           
            cv2.putText(canvas, str(i+1), all_peaks[i][all_peaks_max_index[i]][0:2], cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors[i], thickness=2, lineType=cv2.LINE_AA) 

    print(input_image)
    cv2.imwrite(input_image.replace(".png", "_processed.jpg"), canvas)
    return canvas



def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dir', type=str, default='sample_images/exp_seq00/', help='input dir')
    # parser.add_argument('--dir', type=str, default='sample_images/exp_seq_18_9_43_10317-10915/', help='input dir')    
    parser.add_argument('--dir', type=str, default='sample_images/exp_seq_03-18-11-27-03_3V/', help='input dir')    
    
    # parser.add_argument('--dir', type=str, default='sample_images/figure/', help='input dir')    
    parser.add_argument('--image', type=str, default='sample_images/test.png', help='input image')
    parser.add_argument('--output', type=str, default='result.png', help='output image')
    parser.add_argument('--model', type=str, default='model/model.h5', help='path to the weights file')

    args = parser.parse_args()
    input_image = args.image
    output = args.output
    keras_weights_file = args.model
    imageDir = args.dir

    print('start processing...')


    KRT4_x = np.zeros((512, 424), dtype=float)
    KRT4_y = np.zeros((512, 424), dtype=float)

    KRT6_x = np.zeros((512, 424), dtype=float)    
    KRT6_y = np.zeros((512, 424), dtype=float)

    KRT8_x = np.zeros((512, 424), dtype=float)
    KRT8_y = np.zeros((512, 424), dtype=float)

    file4 = open(imageDir + "../D4.pmm", 'r')
    file6 = open(imageDir + "../D6.pmm", 'r')
    file8 = open(imageDir + "../D8.pmm", 'r')

    text4 = file4.read(-1)
    lines4 = re.split('\n|\r', text4)

    for i in range(3, len(lines4)):
        textValues = re.split(' |\r', lines4[i])
        for j in range(0, len(textValues)-1):
            # print(textValues[j])
            xy = re.split(';| |\n|\t|\r', textValues[j])
            # xy = textValues[j].split(";\n ")
            KRT4_x[j][i-3] = float(xy[0])
            KRT4_y[j][i-3] = float(xy[1])

    text6 = file6.read(-1)
    lines6 = re.split('\n|\r', text6)

    for i in range(3, len(lines6)):
        textValues = re.split(' |\r', lines6[i])
        for j in range(0, len(textValues)-1):
            # print(textValues[j])
            xy = re.split(';| |\n|\t|\r', textValues[j])
            # xy = textValues[j].split(";\n ")
            KRT6_x[j][i-3] = float(xy[0])
            KRT6_y[j][i-3] = float(xy[1])

    text8 = file8.read(-1)
    lines8 = re.split('\n|\r', text8)

    for i in range(3, len(lines8)):
        textValues = re.split(' |\r', lines8[i])
        for j in range(0, len(textValues)-1):
            # print(textValues[j])
            xy = re.split(';| |\n|\t|\r', textValues[j])
            # xy = textValues[j].split(";\n ")
            KRT8_x[j][i-3] = float(xy[0])
            KRT8_y[j][i-3] = float(xy[1])
                        # for (int j = 3; j < lines.Length; ++j)
                        # {
                        #     var values = lines[j].Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                        #     for (int k = 0; k < values.Length; ++k)
                        #     {
                        #         var xy = values[k].Split(new char[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
                        #         // this.loader.KRT[i][k + (j - 3)*512] = new Vector2(float.Parse(xy[0].Replace(".", ",")), float.Parse(xy[1].Replace(".", ",")));
                        #         this.loader.KRT[i][k + (j - 3)*512] = new Vector2(float.Parse(xy[0]), float.Parse(xy[1]));
                        #     }                            
                        # }
                        
    Ext4 = np.zeros((4, 4), dtype=float)
    Ext6 = np.zeros((4, 4), dtype=float)
    Ext8 = np.zeros((4, 4), dtype=float)
    tempExt = np.zeros((4, 4), dtype=float)
  

    file4extrinsics = open(imageDir + "../D4.extrinsics", 'r')
    file6extrinsics = open(imageDir + "../D6.extrinsics", 'r')
    file8extrinsics = open(imageDir + "../D8.extrinsics", 'r')

    ext_text4 = file4extrinsics.read(-1)
    lines_ext4 = re.split('\n|\r', ext_text4)

    ext_text6 = file6extrinsics.read(-1)
    lines_ext6 = re.split('\n|\r', ext_text6)

    ext_text8 = file8extrinsics.read(-1)
    lines_ext8 = re.split('\n|\r', ext_text8)

   
    for i in range(0, len(lines_ext4)):
        textValues = re.split(' |\r', lines_ext4[i])
        for j in range(0, len(textValues)-1):
            # xyz = re.split(';| |\n|\t|\r', textValues[j])
            tempExt[i][j] = float(textValues[j])

    # this.loader.Extrinsics[i] =
    #             new OpenTK.Matrix4(extrinsics[0][0], extrinsics[0][1], extrinsics[0][2], extrinsics[3][0] / divider,
    #                                extrinsics[2][0], extrinsics[2][1], extrinsics[2][2], extrinsics[3][2] / divider,
    #                                -extrinsics[1][0], -extrinsics[1][1], -extrinsics[1][2], -extrinsics[3][1] / divider,
    #                                0, 0, 0, 1);

    Ext4[:][0] = tempExt[:][0]
    Ext4[:][1] = tempExt[:][2]
    Ext4[:][2] = -tempExt[:][1]

    Ext4[0][3] = tempExt[3][0] / 1000.0
    Ext4[1][3] = tempExt[3][2] / 1000.0
    Ext4[2][3] = -tempExt[3][1] / 1000.0

    Ext4[3][3] = 1



    # Ext4[3] = (Ext4[0][3] / 1000.0, Ext4[1][3], Ext4[2][3], 1)
    
    # Ext4[0][3] = 0
    # Ext4[1][3] = 0
    # Ext4[2][3] = 0

    tempExt = np.zeros((4, 4), dtype=float)

    print("3: " + str(Ext4[0][3]))

    for i in range(0, len(lines_ext6)):
        textValues = re.split(' |\r', lines_ext6[i])
        for j in range(0, len(textValues)-1):
            # xyz = re.split(';| |\n|\t|\r', textValues[j])
            tempExt[i][j] = float(textValues[j])
  
    Ext6[:][0] = tempExt[:][0]
    Ext6[:][1] = tempExt[:][2]
    Ext6[:][2] = -tempExt[:][1]

    Ext6[0][3] = tempExt[3][0] / 1000.0
    Ext6[1][3] = tempExt[3][2] / 1000.0
    Ext6[2][3] = -tempExt[3][1] / 1000.0

    Ext6[3][3] = 1
    # Ext6[3] = (Ext6[0][3] / 1000.0, Ext6[1][3], Ext6[2][3], 1)
    
    # Ext6[0][3] = 0
    # Ext6[1][3] = 0
    # Ext6[2][3] = 0

    tempExt = np.zeros((4, 4), dtype=float)
    
    for i in range(0, len(lines_ext8)):
        textValues = re.split(' |\r', lines_ext8[i])
        for j in range(0, len(textValues)-1):
            # xyz = re.split(';| |\n|\t|\r', textValues[j])
            tempExt[i][j] = float(textValues[j])
  
    Ext8[:][0] = tempExt[:][0]
    Ext8[:][1] = tempExt[:][2]
    Ext8[:][2] = -tempExt[:][1]

    Ext8[0][3] = tempExt[3][0] / 1000.0
    Ext8[1][3] = tempExt[3][2] / 1000.0
    Ext8[2][3] = -tempExt[3][1] / 1000.0

    Ext8[3][3] = 1

    # Ext8[3] = (Ext8[0][3] / 1000.0, Ext8[1][3], Ext8[2][3], 1)
    
    # Ext8[0][3] = 0
    # Ext8[1][3] = 0
    # Ext8[2][3] = 0


    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()
    multiplier = [x * model_params['boxsize'] / 424 for x in params['scale_search']]
    imageFiles = []

    if (imageDir):
        
        os.chdir(imageDir)
        imageFiles = glob.glob("*mc_blob.png")
        frameIndex = 0
        for inputImage in imageFiles:
            file_3d = open(inputImage.replace(".png", "_reflectors.txt"), 'w')
            
            # generate image with body parts
            canvas = process(inputImage, params, model_params)
            # cv2.destroyAllWindows()
         
            # cv2.imwrite(inputImage.replace(".png", '_out.png'), canvas)
            # cv2.imshow("MCuDaRT", canvas)
            frameIndex = frameIndex + 1
            # cv2.waitKey(0)

        file_3d.close()
    else:
        tic = time.time()
        # generate image with body parts
        canvas = process(input_image, params, model_params)

        toc = time.time()
        print ('processing time is %.5f' % (toc - tic))

        cv2.imwrite(output, canvas)
        cv2.imshow("MCuDaRT", canvas)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()



