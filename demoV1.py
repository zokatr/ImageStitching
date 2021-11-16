import numpy as np
import glob, os
from enum import Enum
import time
import cv2
from cv2.xfeatures2d import matchGMS
import matplotlib.pyplot as plt


def preprocessImg(wf,mf,area,width,height):

    # Define Search Area
    x1 = area[0]
    y1 = area[1]
    x2 = x1+width
    y2 = y1+height
    
    # Crop wf  and resize mf
    img1 = wf[y1:y2, x1:x2]

    # Resize mf & wf
    w = 800
    h = 500
    dim = (w, h)
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(mf, dim, interpolation = cv2.INTER_AREA)

    return img1,img2


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
    M, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, 6.0)
    matchesMask = mask.ravel().tolist()

    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    dst = cv2.perspectiveTransform(pts,M)
    warped_img = cv2.warpPerspective(img2, M, (w, h))
    overlayed_img = cv2.addWeighted(img1,0.6,warped_img,0.4,0)

    if title == "ORB":
        rect = cv2.polylines(img1,[np.int64(dst)],True,(0,255,0),1, cv2.LINE_AA)
        overlayed_img = cv2.polylines(overlayed_img,[np.int64(dst)],True,(0,255,0),1, cv2.LINE_AA)
    else:
        rect = cv2.polylines(img1,[np.int64(dst)],True,(0,255,255),1, cv2.LINE_AA)
    

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    output = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)

    cv2.imshow(title,overlayed_img)
    cv2.imshow("matches",output)

def matchORB(img1,img2,last_call):
    
    # ORB Feature Detect & Match
    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = bf.match(des1, des2)

    #matches_gms = sorted(matches_all, key = lambda x : x.distance)[0:20]
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=True, withRotation=True, thresholdFactor=6)
    
    # Draw Outputs
    if last_call:
        visualizePosition(img1,img2,kp1,kp2,matches_gms,"ORB")

    print("GMS COUNT  : ",len(matches_gms))
    score = len(matches_gms)
    return score


def matchSIFT(img1,img2,last_call):
    
    #SHIFT Detect & Match
    descriptor = cv2.xfeatures2d.SIFT_create()
    kp1_s, des1_s = descriptor.detectAndCompute(img1, None)
    kp2_s, des2_s = descriptor.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1_s,des2_s,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)

    # Draw Outputs
    if last_call:
        visualizePosition(img1,img2,kp1_s,kp2_s,good,"SHIFT")
    print("SIFT COUNT : ",len(good))

    return len(good)


if __name__ == '__main__':

    start = time.time()
    search_areas = [[95,1945],[510,1990],[1015,1855],[1500,1700],[1860,1855],[2355,1850],[2950,1850],[3425,1820],[3920,1830],[4160,1775],[4650,1765],[5150,1775]]
    width,height = 1600,1000
    
    # Read Img
    wf = cv2.imread("./wf.jpg")
    files = os.listdir("./")
    os.chdir("./")
    for file in glob.glob("*.JPG"):

        print("file ",file)
        best_score = -999
        best_area_index = -999

        if file[:-4] != "wf":
            mf = cv2.imread(file)
            for i,area in enumerate(search_areas):

                img1, img2 = preprocessImg(wf,mf,area,width,height)
                score = matchORB(img1,img2,False)
                #score = matchSIFT(img1,img2,False)
                if score>best_score:
                    best_score = score
                    best_area = area


            
            img1, img2 = preprocessImg(wf,mf,best_area,width,height)
            score = matchORB(img1,img2,True)
            #score = matchSIFT(img1,img2,True)
            
            #del search_areas[best_area_index]
            cv2.waitKey(0)



    
    # Draw Outputs
    #visualizePosition(img1,img2,kp1,kp2,matches_gms,"ORB")
    #visualizePosition(img1,img2,kp1_s,kp2_s,good,"SHIFT")
  
    end = time.time()
    print('Total Time: ', end-start, 'seconds')
    
    #cv2.imshow("show", output)
    cv2.waitKey(0)
