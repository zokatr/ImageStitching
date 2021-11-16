import numpy as np
import glob, os
from enum import Enum
import time
import cv2
from cv2.xfeatures2d import matchGMS
import matplotlib.pyplot as plt



def visualizePosition(img1,img2,kp1,kp2,matches,title):
    """
    img1 : wide field
    img2 : medium field
    title: algorithm name
    Draw rectangle
    Overlay the medium img
    """

    h,w,d = img1.shape
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    dst = cv2.perspectiveTransform(pts,M)
    warped_img = cv2.warpPerspective(img2, M, (w, h))
    overlayed_img = cv2.addWeighted(img1,0.7,warped_img,0.3,0)

    if title == "ORB":
        rect = cv2.polylines(img1,[np.int64(dst)],True,(0,255,0),1, cv2.LINE_AA)
    else:
        rect = cv2.polylines(img1,[np.int64(dst)],True,(0,255,255),1, cv2.LINE_AA)
    

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    output = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)

    cv2.imshow(title,overlayed_img)
    cv2.imshow("matches",output)



if __name__ == '__main__':
    start = time.time()
    # Read Img
    img1 = cv2.imread("./wf.jpg")
    img2 = cv2.imread("./mf00.jpg")



    print ("wf shape : ",img1.shape)
    print ("mf shape : ",img2.shape)
    # Define Search Area
    x1 = 0
    y1 = 2000
    x2 = x1+1600
    y2 = y1+1000
    
    # Crop wf  and resize mf
    img1 = img1[y1:y2, x1:x2]

    # Resize mf & wf
    width = 800
    height = 500
    dim = (width, height)
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    
    # ORB Feature Detect & Match
    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING) # NORM_HAMMING
    matches_all = bf.match(des1, des2)
    # The matches with shorter distance are the ones we want.
    #matches_gms = sorted(matches_all, key = lambda x : x.distance)[0:20]
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=True, withRotation=False, thresholdFactor=6)
    print("GMS COUNT  : ",len(matches_gms))

    #SHIFT Detect & Match
    descriptor = cv2.xfeatures2d.SIFT_create()
    kp1_s, des1_s = descriptor.detectAndCompute(img1, None)
    kp2_s, des2_s = descriptor.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1_s,des2_s,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)

    print("SIFT COUNT : ",len(good))
    

    # Draw Outputs
    visualizePosition(img1,img2,kp1_s,kp2_s,good,"SHIFT")
    visualizePosition(img1,img2,kp1,kp2,matches_gms,"ORB")
  
    end = time.time()
    print('Total Time: ', end-start, 'seconds')
    
    #cv2.imshow("show", output)
    cv2.waitKey(0)
