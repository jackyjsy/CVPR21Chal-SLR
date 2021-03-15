import numpy as np
import math

import cv2

def pose_process(coords, hms):
    # hms: num_joints x h x w numpy array
    # coords: num_joints x 2 numpy array
    hm_w = hms.shape[2]
    hm_h = hms.shape[1]
    # score = np.zeros((hms.shape[0],1), dtype=np.float32)

    # print(hm_w, ', ', hm_h)
    for p in range(coords.shape[0]):
        hm = hms[p]
        
        px = int(math.floor(coords[p][0] + 0.5))
        py = int(math.floor(coords[p][1] + 0.5))
        # score[p] = hm[py, px]
        coords[p][2] = hm[py, px]
        if 1 < px < hm_w -1 and 1 < py < hm_h - 1:
            diff = np.array(
                [hm[py][px+1] - hm[py][px-1], hm[py+1][px] - hm[py-1][px]]
            )
            # print(px, ', ', py)
            # print(diff[0], ', ', diff[1])
            coords[p][:2] += np.sign(diff) * .25
    return coords

def plot_pose(img,result,scale=(1.0,1.0)):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    PINK = (203, 192, 255)
    unshown_pts = [0,1,2,3,4,11,12,13,14,15,16,17,18,19,20,21,22]
    l_pair = [         #v3 web
        #coco pose
        
        (5, 7),# R shoulder - R elbow
        (7, 9),# R elbow - R wrist
        
        (6, 8), # M shoulder - L shoulder
        (8, 10),# L shoulder - L elbow
        (5, 6),# L elbow - L wrist
        
#====================================
#=========feet ===================

#========================================
#=========face===========================
        (23,24),#contour
         (24,25),
        (25,26),
        (26,27),
        (27,28),
        (28,29),
        (29,30),
        (30,31),
        (31,32),
        (32,33),
        (33,34),
        (34,35),
        (35,36),
        (36,37),
        (37,38),
        (38,39),
        
        (40,41),#R eye brows
        (41,42),
        (42,43),
        (43,44),
        (45,46),#L eye brows
        (46,47),
        (47,48),
        (48,49),
        
        (50,51), #nose
        (51,52),
        (52,53),
        (54,55),
        (55,56),
        (56,57),
        (57,58),
        
        (59,60),# R eye
        (60,61),
        (61,62),
        (62,63),
        (63,64),
        (64,59),
        (65,66),#L eye
        (66,67),
        (67,68),
        (68,69),
        (69,70),
        (70,65),
        
        (71,72),#out mouth
        (72,73),
        (73,74),
        (74,75),
        (75,76),
        (76,77),
        (77,78),
        (78,79),
        (79,80),
        (80,81),
        (81,82),
        (82,71),
        
        (83,84),#in mouth
        (84,85),
        (85,86),
        (86,87),
        (87,88),
        (88,89),
        (89,90),
        (90,83),
#==================================================
#=============hands================================
        (9,91),#L wrist - L hand
        (91,92),# L thumb
        (92,93),
        (93,94),
        (94,95),
        (91,96),#L index finger
        (96,97),
        (97,98),
        (98,99),
        (91,100), #L mid finger
        (100,101),
        (101,102),
        (102,103),
        (91,104), # L ring finger
        (104,105),
        (105,106),
        (106,107),
        (91,108), #L little finger
        (108,109),
        (109,110),
        (110,111),
        
        (10,112), # R wrist - R hand
        (112,113),#R thumb
        (113,114),
        (114,115),
        (115,116),
        (112,117),#R index finger
        (117,118),
        (118,119),
        (119,120),
        (112,121),#R mid finger
        (121,122),
        (122,123),
        (123,124),
        (112,125),#R ring finger
        (125,126),
        (126,127),
        (127,128),
        (112,129), #R little finger
        (129,130),
        (130,131),
        (131,132),
    ]
    p_color = [GREEN, BLUE, BLUE, BLUE, BLUE, YELLOW, ORANGE, YELLOW, ORANGE,
                YELLOW, ORANGE, PINK, RED, PINK, RED, PINK, RED]
    p_color = [GREEN, BLUE, BLUE, BLUE, BLUE, YELLOW, ORANGE, YELLOW, ORANGE,
                YELLOW, ORANGE, PINK, RED, PINK, RED, PINK]

    part_line = {}
    kp_preds = result[:, :2]
    kp_scores = result[:, 2]
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #print(kp_scores.size())
    # Draw keypoints
    for n in range(kp_scores.shape[0]):
        if kp_scores[n] <= 0.1 or n in unshown_pts:
            continue
        cor_x, cor_y = int(round(kp_preds[n, 0] * scale[0])), int(round(kp_preds[n, 1] * scale[1]))
        part_line[n] = (cor_x, cor_y)
        cv2.circle(img, (cor_x, cor_y), 2, (0,0,255), -1)
    # Draw limbs
    for start_p, end_p in l_pair:
        if start_p in part_line and end_p in part_line:
            start_p = part_line[start_p]
            end_p = part_line[end_p]
            cv2.line(img, start_p, end_p, (0,255,0), 2)
    return img