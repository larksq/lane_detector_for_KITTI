import cv2
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
import cubic
from sklearn.preprocessing import MinMaxScaler
import math
from scipy import ndimage
import scipy.cluster.hierarchy as hcluster
import random
import os, shutil
import time

DEFAULT_LINE_LIST = ['CD']
DEFAULT_LINE_INFO = [[(770, 170), 150, 50, 0]]

class LineDetector:

    # the list of current lines information at the starting point
    # C = Curb, S = Solid (white) line, D = Dashed (white) line
    # SS = Straight Solid line, CD = Curved Dashed line, CS = Curved Solid line ..
    # each line should be a detectable visual unit

    # line info: [mid point(x, y) at bottom for initial detect window,
    #             detect window height,
    #             detect window width,
    #             initial angel (in radians)]

    def __init__(self,
                 min_over_hough = 50,
                 min_length = 56/3,
                 max_length = 56*1.2,
                 image_dimension = (800, 400),
                 dash_interval_pxl = 90,
                 line_list= DEFAULT_LINE_LIST,
                 line_info= DEFAULT_LINE_INFO,
                 lane_width = 53,
                 initial_angle = 0,
                 max_turning_angle = [-100, 100],
                 current_image_name = 'um_000000',
                 current_image_path= "../../../../KITTI/data_road/transformed/",
                 vis_folder_prefix = 'visualization/',
                 step_window= False,
                 visualization = False
                 ):

        self.min_over_hough = min_over_hough
        self.min_length = min_length
        self.max_length= max_length
        self.image_dimension= image_dimension
        self.dash_interval_pxl= dash_interval_pxl # the interval + one line length, 0-300 is legal value for window_h=200
        self.line_list= line_list
        self.line_info= line_info
        self.lane_width= lane_width # used for solid line mode
        self.initial_angle= initial_angle
        self.current_image_name= current_image_name
        self.current_image_path= current_image_path
        self.vis_folder_prefix = vis_folder_prefix
        assert len(self.line_info) == len(self.line_list)
        self.step_window= step_window
        self.visualization = visualization
        self.max_turning_angle = max_turning_angle


    @staticmethod
    def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.

        point = (3, 4)
        origin = (2, 2)
        rotate(origin, point, math.radians(10))
        (2.6375113976783475, 4.143263683691346)
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy


    @staticmethod
    def prepare_visualization_folder(vis_folder):
        # clear visualization folder
        for filename in os.listdir(vis_folder):
            file_path = os.path.join(vis_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    def detect_and_save(self):

        start_time = time.perf_counter()

        vis_folder = self.vis_folder_prefix+str(self.current_image_name)
        self.prepare_visualization_folder(vis_folder=vis_folder)
        img = cv2.imread(self.current_image_path+str(self.current_image_name)+".png")
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)

        normal_k_v_7 = np.array([[-10, 1, 0, 1, +10],
                              [0,-10, 1, +10, 0],
                              [0, -10, 1, +10, 0],
                              [0, -10, 1, +10, 0],
                              [-10, 1, 0, 1, +10]])

        grad = signal.convolve2d(y, normal_k_v_7, boundary='symm', mode='same')
        scaler = MinMaxScaler()
        scaler.fit(np.array(grad).reshape(-1,1))
        grad_n = scaler.transform(grad)*255
        origin = (self.image_dimension[0]/2, self.image_dimension[1]/2)
        line_detection_results = []

        for i, line_info_list in enumerate(self.line_info): # for each line
            print("searching no.", i, " line :", self.line_list[i])
            bottom_middle_end, window_h, window_w, initial_angle = line_info_list
            if self.initial_angle == 0:
                current_mid_end = bottom_middle_end
            else: # rotate based on middle end of the image
                d_grad = np.zeros((self.image_dimension[0]*2, self.image_dimension[1]))
                d_grad[0:self.image_dimension[0],:] = grad_n
                angle_in_degree = self.initial_angle *180 / np.pi
                grad_n = ndimage.rotate(input=np.array(d_grad), angle=angle_in_degree, reshape=False)[0:self.image_dimension[0],:]
                cv2.imwrite(os.path.join(vis_folder, "init_rotated.png"), grad_n)
                bottom_middle_end = self.rotate(origin = (self.image_dimension[0], int(self.image_dimension[1]/2)),
                                                point = bottom_middle_end,
                                                angle = self.initial_angle)
                bottom_middle_end = (int(bottom_middle_end[0]), int(bottom_middle_end[1]))
                current_mid_end = bottom_middle_end

            mid_x, mid_y = bottom_middle_end
            window_coor = [] # this is for visualization windows
            line_type = self.line_list[i]
            line_vis_folder = os.path.join(vis_folder, line_type)
            if self.visualization:
                if not os.path.exists(line_vis_folder):
                    os.mkdir(line_vis_folder)
            if line_type == 'SS':
                window1 = grad_n[:mid_x, mid_y-int(window_w/2): mid_y+int(window_w/2)]
                window_h = mid_x
            else:
                window1 = grad_n[mid_x-window_h: mid_x, mid_y-int(window_w/2): mid_y+int(window_w/2)]
            line_ended = False
            # line_detected = False
            iteration = 0
            angle_to_clockwise_current = 0
            angle_to_clockwise_total = 0
            final_mask_for_the_line = np.zeros(grad_n.shape)

            while not line_ended: # for each window
                iteration += 1
                if self.visualization:
                    cv2.imwrite(os.path.join(line_vis_folder, 'current_window'+str(iteration)+'.png'),window1)
                cluster_and_spline = True
                clusters_no = 0
                max_line_angle = 0
                index_for_mask = ([], [])
                # re-normalize in the window
                scaler_2 = MinMaxScaler()
                scaler_2.fit(np.array(window1).reshape(-1,1))
                window1_n = scaler_2.transform(window1)*255
                if self.visualization:
                    cv2.imwrite(os.path.join(line_vis_folder, 'window_n.png'),window1_n)

                # erosion will enhance the dark area
                # print("test: ",np.mean(window1_n), min(220, 140/110*np.mean(window1_n)), np.max(window1_n)) # inspect clip
                min_threshold = min(220, 140/110*np.mean(window1_n))
                window_clip = np.clip(window1_n, min_threshold, 255) - min_threshold
                window_clip = window_clip / (255 - min_threshold) * 255
                cv2.imwrite(os.path.join(line_vis_folder, 'clip.png'),window_clip)
                if abs(angle_to_clockwise_current)>0.1 or abs(self.initial_angle)>0.1:
                    print("sharp turns, skipping erosion")
                    erosion = window_clip
                else:
                    kernel = np.ones((5,1),np.uint8)
                    erosion = cv2.erode(window_clip,kernel,iterations = 3)
                if self.visualization:
                    cv2.imwrite(os.path.join(line_vis_folder, 'bin.png'),erosion)

                # kernel = np.ones((9,9),np.uint8) # backup kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
                erosion = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

                # clustering lines
                img_np = np.array(erosion)
                img_idx_tp = np.where(img_np > self.min_over_hough)
                if len(img_idx_tp[0]) < 5:
                    print("little or no pixel above minimal hough")
                else:
                    img_idx_list = [img_idx_tp[0], img_idx_tp[1]]
                    img_idx_arr = np.transpose(np.array(img_idx_list))
                    # the minimal distance in pixel for clustering, 4 is the experiential value
                    d_thresh = 4
                    clusters = hcluster.fclusterdata(img_idx_arr, d_thresh, criterion="distance")
                    clusters_no = np.max(clusters)
                    print("total cluster: ",clusters_no)

                h, w = img_np.shape
                if clusters_no > 10:
                    print("too noisy, quite detecting lines")
                    cluster_and_spline = False
                elif clusters_no == 0:
                    print("no cluster result, quite detecting lines")
                    cluster_and_spline = False
                else:
                    # for visualization
                    copy = np.zeros((h, w, 3))
                    color_pallet = []
                    for i in range(0, clusters_no):
                        color_pallet.append([random.random()*255, random.random()*255, random.random()*255])
                    for i, pos in enumerate(img_idx_arr):
                        x, y = pos
                        group_id = clusters[i]
                        copy[x, y] = color_pallet[group_id-1]
                    if self.visualization:
                        cv2.imwrite(os.path.join(line_vis_folder, 'cluster_before_filter.png'),copy)
                    # end of visualization

                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))
                copy_1 = np.zeros((h, w, 3))
                copy_2 = np.zeros((h, w, 3))
                max_x_span = 0
                max_line = np.array([])
                max_x_center = 0

                if cluster_and_spline:
                    # get the lines
                    # this init is for final marking
                    x_list_before_r = []
                    y_list_before_r = []

                    for c_no in range(0, clusters_no):
                        # for each line (cluster)
                        zero_img = np.zeros(img_np.shape)
                        pos_list = []
                        for i, pos in enumerate(img_idx_arr):
                            if clusters[i] == c_no+1:
                                x, y = pos
                                pos_list.append(pos)
                                zero_img[x, y] = img_np[x, y]

                        x_list, y_list = np.where(zero_img > self.min_over_hough)
                        if len(x_list) > 0:
                            x_span = np.max(x_list) - np.min(x_list)
                            y_span = np.max(y_list) - np.min(y_list)
                            if self.max_length > x_span > self.min_length or y_span > self.min_length:
                                # filter small and weak lines
                                # print("valid:", c_no, x_span, y_span)

                                # visulization after filter
                                for pos in pos_list:
                                    x, y = pos
                                    copy_1[x, y] = color_pallet[c_no]

                                # 1. randomly select a few points (deprecated)
                                # all_points_num = len(x_list)
                                # random select makes line bend because choosing points varying horizontally
                                # if all_points_num > SPLINE_SEEDS_NUMBER:
                                #     for i in range(0, SPLINE_SEEDS_NUMBER):
                                #         seed = int(random.random()*(all_points_num-1))
                                #         x_spline.append(x_list[seed])
                                #         y_spline.append(y_list[seed])
                                # else:
                                #     print("points are fewer than 50, continue")
                                #     continue

                                # choose all points to spline
                                x_spline = x_list
                                y_spline = y_list
                                x_y_arr = np.array([x_spline, y_spline])
                                x_y_arr_1 = np.transpose(x_y_arr)
                                x_y_arr_sorted = np.sort(x_y_arr_1.copy().view('i8,i8'), order=['f0'], axis=0).view(np.int)
                                x_y_arr_trans = np.transpose(x_y_arr_sorted)

                                # 2. fit the line
                                # x_spline = np.linspace(0, SPLINE_SEEDS_NUMBER, num=SPLINE_SEEDS_NUMBER, endpoint=False)
                                # f = interp1d(x_spline, y_spline)
                                # tck = interpolate.splrep(x_y_arr_trans[0], x_y_arr_trans[1], s=0)
                                # xnew = np.arange(0, 2*np.pi, np.pi/50)
                                # ynew = interpolate.splev(xnew, tck, der=0)
                                # f2 = interp1d(x_spline, y_spline, kind='cubic')

                                # for dashed line use linear regression
                                if line_type in ['SS', 'CS']:
                                    # The number of knots can be used to control the amount of smoothness
                                    x, y = x_y_arr_trans
                                    model_6 = cubic.get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=6)
                                    # model_15 = cubic.get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=15)
                                    xnew_arr = np.array(range(0, h)) # np.linspace(0, h, num=x_span, endpoint=False)
                                    y_new = model_6.predict(xnew_arr)
                                    diff = (y_new[5] - y_new[45])/40.0
                                    the_angle = np.arctan2(diff, 1)
                                elif line_type in ['SD', 'CD']:
                                    x, y = x_y_arr_trans
                                    X = x.reshape(-1,1)
                                    reg = LinearRegression().fit(X, y)
                                    print("linear regression: ", reg.score(X, y), reg.coef_, reg.intercept_)
                                    the_angle = reg.coef_[0] * -1
                                    xnew_arr = np.array(range(0, h)) # np.linspace(0, h, num=x_span, endpoint=False)
                                    y_new = reg.predict(xnew_arr.reshape(-1,1))

                                # 4. check the line's validity
                                if not self.max_turning_angle[1] >= the_angle * 180 / np.pi >= self.max_turning_angle[0]:
                                    print("large line angle, continue", the_angle * 180 / np.pi)
                                    continue
                                print("test: ", abs(the_angle) * 180 / np.pi)

                                # 5. mark pixels on the image before rotation
                                x_offset = current_mid_end[0] - window_h
                                y_offset = current_mid_end[1] - int(window_w/2)
                                if line_type == 'SS':
                                    for i, ynew_y in enumerate(y_new):
                                        if 0 < ynew_y < w-1:
                                            copy_2[int(i), int(ynew_y)] = np.array(color_pallet[c_no]) * y_span / 3
                                    # for SS, we keep all spline for marking
                                    non_zero_line_pixels = (copy_2[:,:,0] + copy_2[:,:,1] + copy_2[:,:,2])/3
                                    x_list_SS, y_list_SS = np.where(non_zero_line_pixels)
                                    for i, x_idx in enumerate(x_list_SS):
                                        y_idx = y_list_SS[i]
                                        previous_angle = angle_to_clockwise_total-angle_to_clockwise_current
                                        point_before_r = self.rotate(origin, (x_idx+x_offset, y_idx+y_offset), previous_angle*-1)
                                        if -1<point_before_r[0]<self.image_dimension[0] and -1<point_before_r[1]<self.image_dimension[1]:
                                            x_list_before_r.append(int(point_before_r[0]))
                                            y_list_before_r.append(int(point_before_r[1]))

                                    index_for_mask = (x_list_before_r, y_list_before_r)
                                    # angle_to_clockwise += delta_angle
                                    max_line = y_new.copy()
                                    continue

                                if x_span > max_x_span:
                                    # print("new max x span")
                                    max_x_span = x_span
                                    copy_2 = np.zeros((h, w, 3))
                                    # 3. redraw the line (only draw the max line)
                                    # NOTE: this is for detection, not only visualization
                                    for i, ynew_y in enumerate(y_new):
                                        if 0 < ynew_y < w-1:
                                            copy_2[int(i), int(ynew_y)] = np.array(color_pallet[c_no]) * y_span / 3
                                    max_line = y_new.copy()
                                    max_line_angle = the_angle
                                    max_x_center = int((np.max(x_list) + np.min(x_list))/2)
                                    # create again every cluster, we only keep the max length cluster
                                    x_list_before_r = []
                                    y_list_before_r = []
                                    # we keep the area of max span
                                    for i, x_idx in enumerate(x_list):#enumerate(np.unique(x_list)):
                                        # y_idx = y_new[x_idx]
                                        y_idx = y_list[i]
                                        point_before_r = self.rotate(origin, (x_idx+x_offset, y_idx+y_offset), angle_to_clockwise_total*-1)
                                        x_list_before_r.append(int(point_before_r[0]))
                                        y_list_before_r.append(int(point_before_r[1]))

                                    for x_idx in range(0, window_h-1):
                                        y_idx = y_new[x_idx]
                                        point_before_r = self.rotate(origin, (x_idx+x_offset, y_idx+y_offset), angle_to_clockwise_total*-1)
                                        x_list_before_r.append(int(point_before_r[0]))
                                        y_list_before_r.append(int(point_before_r[1]))

                                    index_for_mask = (x_list_before_r, y_list_before_r)

                                else:
                                    continue

                else:
                    print("skipping line detection")

                if self.visualization:
                    cv2.imwrite(os.path.join(line_vis_folder, 'cluster_after_filter.png'),copy_1)
                    cv2.imwrite(os.path.join(line_vis_folder, 'splines.png'),copy_2)

                if len(max_line) < 20:
                    print("no valid line detected")
                    if line_type == 'SS':
                        print("solid straight line not found")
                        line_ended = True
                        continue
                    middle_end = (current_mid_end[0]-window_h+5, current_mid_end[1])
                    angle_to_clockwise_current = 0
                else:
                    # line_detected = True
                    angle_to_clockwise_total += max_line_angle
                    angle_to_clockwise_current = max_line_angle
                    if line_type == 'CS':
                        print("CS angle: ",angle_to_clockwise_total * 180 / np.pi, angle_to_clockwise_current * 180 / np.pi)
                        middle_end = (current_mid_end[0]-window_h+5, max_line[5]+current_mid_end[1]-int(window_w/2))
                        final_mask_for_the_line[index_for_mask] = 255

                    elif line_type in ['SD', 'CD']:
                        # for SD and CD, we slide the new window based on current detection center
                        # the angle is already plused in linear regression
                        target_x_bottom = max_x_center-self.dash_interval_pxl+int(window_h/2)
                        # # WARNING: might went out of window index and crash if dash interval and window_h are changed
                        # diff = (max_line[target_x_bottom] - max_line[target_x_bottom+10])/10.0
                        # angle_to_clockwise += np.arctan2(diff, 1) # in radians
                        print("SD/CD angle: ",angle_to_clockwise_total * 180 / np.pi, angle_to_clockwise_current * 180 / np.pi)
                        middle_end = (current_mid_end[0]-window_h+target_x_bottom, max_line[target_x_bottom]+current_mid_end[1]-int(window_w/2))
                        final_mask_for_the_line[index_for_mask] = 255
                    elif line_type == 'SS':
                        line_ended = True
                        final_mask_for_the_line[index_for_mask] = 255
                        window_coor.append([(mid_y-int(window_w/2), 0), (mid_y+int(window_w/2), mid_x)])
                        continue
                    else:
                        print("ERROR: illegal line type ", line_type)

                # moving window and visualization (only for non SS line)
                middle_end_after_rotation = self.rotate(origin, middle_end, angle_to_clockwise_current)
                new_x_range = (int(middle_end_after_rotation[0] - window_h), int(middle_end_after_rotation[0]))
                new_y_range = (int(middle_end_after_rotation[1] - window_w/2), int(middle_end_after_rotation[1] + window_w/2))

                # for final visual mask
                middle_end_original_coor = self.rotate(origin, middle_end, (angle_to_clockwise_total - angle_to_clockwise_current)*-1)
                before_x_range = (int(middle_end_original_coor[0] - window_h), int(middle_end_original_coor[0]))
                before_y_range = (int(middle_end_original_coor[1] - window_w/2), int(middle_end_original_coor[1] + window_w/2))

                # print("inspect: ", origin, middle_end, middle_end_after_rotation, new_x_range, new_y_range, angle_to_clockwise_current, angle_to_clockwise_total)

                if new_x_range[0] < 0 or new_x_range[1] > self.image_dimension[0]:
                    print("illegal next window, x out of image bound")
                    line_ended = True
                    continue

                if new_y_range[0] < 0 or new_y_range[1] > self.image_dimension[1]:
                    print("illegal next window, y out of image bound")
                    line_ended = True
                    continue

                # positive is conter clockwise
                angle_in_degree = angle_to_clockwise_total *180 / np.pi
                grad_n_r = ndimage.rotate(input=np.array(grad_n), angle=angle_in_degree, reshape=False)
                cv2.imwrite(os.path.join(line_vis_folder, 'after_rotate.png'),grad_n_r)
                window1 = grad_n_r[new_x_range[0]:new_x_range[1], new_y_range[0]:new_y_range[1]]

                if np.min(window1) <=0:
                    print("illegal next window at image edge, caused by rotation")
                    line_ended = True
                    continue

                window_coor.append([(before_y_range[0], before_x_range[0]),(before_y_range[1], before_x_range[1])]) # for opencv goes w,h
                current_mid_end = middle_end_after_rotation

                # check window by window
                if self.step_window:
                    while(True):
                        print("pausing, press anykey to continue to next window")
                        a = input()
                        if a:
                            break

            # end loop of the line in each window
            if self.visualization:
            # for visualization windows
                final_mask_for_the_line_vis = cv2.cvtColor(np.array(final_mask_for_the_line, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
                for cood in window_coor:
                    lt, rb = cood
                    print(cood, lt, rb)
                    cv2.rectangle(final_mask_for_the_line_vis,lt,rb,(255,0,0),1)
                cv2.imwrite(os.path.join(line_vis_folder, 'mask.png'),final_mask_for_the_line_vis)
            # end of visual

            # loop and connect detection failure windows
            start_point = []
            end_point = []
            last_x_has_line = True

            for x in range(0, self.image_dimension[0]-1):
                x = self.image_dimension[0] - x - 2
                if np.max(final_mask_for_the_line[x,:]) > 200:
                    if not last_x_has_line:
                        all_y = np.where(final_mask_for_the_line[x,:]>200)[0]
                        end_point.append((x, int(np.mean(all_y))))
                        last_x_has_line = True
                else:
                    if last_x_has_line:
                        all_y = np.where(final_mask_for_the_line[x+1,:]>200)[0]
                        if len(all_y) <= 0:
                            continue
                        start_point.append((x, int(np.mean(all_y))))
                        last_x_has_line = False

            # ending with an additional useless starting point
            # connect starting points and ending points
            for i, end_ptx in enumerate(end_point):
                end_x, end_y = end_ptx
                start_x, start_y = start_point[i]
                final_mask_for_the_line = cv2.line(final_mask_for_the_line.astype(np.uint8), (start_y, start_x),(end_y, end_x),(255,0,0),1)

            if self.visualization:
                cv2.imwrite(os.path.join(line_vis_folder, 'undetected_lines_drawn.png'),final_mask_for_the_line)

            # spline the final line
            x_spline, y_spline = np.where(final_mask_for_the_line>200)
            x_y_arr = np.array([x_spline, y_spline])
            x_y_arr_1 = np.transpose(x_y_arr)
            x_y_arr_sorted = np.sort(x_y_arr_1.copy().view('i8,i8'), order=['f0'], axis=0).view(np.int)
            x_y_arr_trans = np.transpose(x_y_arr_sorted)
            x, y = x_y_arr_trans
            # model_6 = cubic.get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=6)
            model_15 = cubic.get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=15)

            # 3. redraw the line
            xnew_arr = np.array(range(0, self.image_dimension[0]))
            copy = np.zeros((self.image_dimension[0], self.image_dimension[1], 3))
            # y_new = interpolate.splev(xnew_arr, tck, der=0)
            y_new = model_15.predict(xnew_arr)
            for i, ynew_y in enumerate(y_new):
                if ynew_y < self.image_dimension[1]-1 and ynew_y > 0:
                    copy[int(i), int(ynew_y)] = np.array([255, 0, 0])

            line_detection_results.append(copy)
            if self.visualization:
                cv2.imwrite(os.path.join(line_vis_folder, 'final_line.png'),copy)


        # end of all lines detection
        # generate the region between the last two
        detection_result = []
        for d in line_detection_results:
            r = (d[:,:,0] + d[:,:,1] + d[:,:,2])/3
            detection_result.append(r)

        # print("test: ", len(detection_result), len(line_detection_results))
        if len(self.line_list) == 1:
            left_idx_x, left_idx_y = np.where(detection_result[0]>0)
            right_idx_y = left_idx_y + self.lane_width
            right_idx_x = left_idx_x.copy()
        elif len(self.line_list) == 2:
            left_idx_x, left_idx_y = np.where(detection_result[0]>0)
            right_idx_x, right_idx_y = np.where(detection_result[1]>0)

        copy = np.zeros((self.image_dimension[0], self.image_dimension[1]))
        overlay = np.zeros((self.image_dimension[0], self.image_dimension[1], 3))
        for r in range(0, self.image_dimension[0]):
            left = 0
            right = 0
            for l_idx, l_x in enumerate(left_idx_x):
                if l_x == r:
                    left = left_idx_y[l_idx]
            for r_idx, r_x in enumerate(right_idx_x):
                if r_x == r:
                    right = right_idx_y[r_idx]

            for c in range(0, self.image_dimension[1]):
                if left < c < right:
                    copy[r, c] = 255
                    overlay[r, c, 2] = 255

        if not self.initial_angle == 0:
            angle_in_degree = self.initial_angle *180 / np.pi
            d_copy = np.zeros((self.image_dimension[0]*2, self.image_dimension[1]))
            d_copy[0:self.image_dimension[0], :] = copy
            d_copy = ndimage.rotate(input=np.array(d_copy), angle=angle_in_degree*-1, reshape=False)
            copy = d_copy[0:self.image_dimension[0], :]

            d_over = np.zeros((self.image_dimension[0]*2, self.image_dimension[1], 3))
            d_over[0:self.image_dimension[0], :] = overlay
            d_over = ndimage.rotate(input=np.array(d_over), angle=angle_in_degree*-1, reshape=False)
            overlay = d_over[0:self.image_dimension[0], :, :]

        if self.visualization:
            dst = cv2.addWeighted(np.array(img, dtype=np.int32), 0.8, np.array(overlay, dtype=np.int32), 0.2, 0.0)
            cv2.imwrite(os.path.join(vis_folder, 'final_seg_layon.png'),dst)

        end_time = time.perf_counter()
        print("finished in ", end_time-start_time," seconds")
        # output the mask result
        cv2.imwrite(os.path.join(vis_folder, 'final_seg.png'),copy)



if __name__ == "__main__":

    detector = LineDetector(visualization=True)

    # Example of how to use on customized data
    #
    # detector = LineDetector(
    #     min_length=80/3,
    #     max_length=80*1.2,
    #     image_dimension=(800, 400),
    #     dash_interval_pxl=135,
    #     max_turning_angle=[-100, 5],
    #     line_list=['CD'],
    #     line_info=[[(770, 172), 150, 50, 0]],
    #     lane_width=50,
    #     initial_angle=0,
    #     visualization=True,
    #     step_window=False,
    #     current_image_name = 'um_000083',
    #     current_image_path= "../../../../KITTI/data_road/transformed/",
    #     vis_folder_prefix = 'visualization/',
    # )

    # Example of how to use customized data from Citiscape
    #
    detector = LineDetector(
        min_length=53/3,
        max_length=53*1.2,
        image_dimension=(1024, 1024),
        dash_interval_pxl=63,
        max_turning_angle=[-20, 20],
        line_list=['CD', 'CD'],
        line_info=[[(942, 451), 80, 50, 0],
                   [(952, 664), 80, 50, 0]],
        lane_width=104,
        initial_angle=0,
        visualization=True,
        step_window=False,
        current_image_name = 'bochum_000000_020673_BEV',
        current_image_path= "../../images/cityscapes/normal/transformed/",
        vis_folder_prefix = 'visualization/',
    )

    detector.detect_and_save()
