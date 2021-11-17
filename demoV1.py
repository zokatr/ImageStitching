import cv2
import time
import glob, os
import numpy as np
from cv2.xfeatures2d import matchGMS



def preprocessImg(wf,mf,area,width,height):
    """
    Crop the area from wide-field image and resize mf and cropped image
    wf: wide-field image
    mf: medium-field image
    return: resized images
    """

    # Define Search Area Coord.
    x1 = area[0]
    y1 = area[1]
    x2 = x1+width
    y2 = y1+height
    
    # Crop wf
    img1 = wf[y1:y2, x1:x2]

    #TODO: w-h should not be harcoded here.
    # Resize mf & wf
    w = 800
    h = 500
    dim = (w, h)
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(mf, dim, interpolation = cv2.INTER_AREA)

    return img1,img2


def visualizePosition(img1,img2,kp1,kp2,matches,title):
    """
    Draw a rectangle to matched area &  Overlay the medium image to wide image
    img1 : wide field
    img2 : medium field
    title: algorithm name
    """

    # Find the homography matrix
    h,w,d = img1.shape
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    H, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, 6.0)
    matchesMask = mask.ravel().tolist()

    # Find the corresponding points for rectange
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)     
    dst = cv2.perspectiveTransform(pts,H)

    # Warp the medium image and overlay to cropped-wide image.
    warped_img = cv2.warpPerspective(img2, H, (w, h))
    overlayed_img = cv2.addWeighted(img1,0.9,warped_img,0.2,0)

    # Draw and show the results
    img_rectangle = cv2.polylines(img1,[np.int64(dst)],True,(0,255,0),1, cv2.LINE_AA)
    overlayed_img = cv2.polylines(overlayed_img,[np.int64(dst)],True,(0,255,0),1, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    output = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
    title = title+" - Overlayed Img"
    cv2.imshow(title,overlayed_img)
    cv2.imshow("matches",output)

def matchORB(img1,img2,last_call):
    """
    Detection & matching the features with ORB
    img1 : wide field
    img2 : medium field
    last_call: visualization flag - True/False
    """
    
    # ORB Feature Detect & Match
    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = bf.match(des1, des2)

    # Filter the matches
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=True, withRotation=False, thresholdFactor=6)
    
    # Draw Outputs
    if last_call:
        visualizePosition(img1,img2,kp1,kp2,matches_gms,"ORB")

    print("ORB-GMS Match count : ",len(matches_gms))

    # To detect the matching area in wide field img, return the count of feature match
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
    print("SIFT Match count : ",len(good))

    return len(good)


if __name__ == '__main__':

    #start = time.time()
    search_areas = [[95,1945],[510,1990],[1015,1855],[1500,1700],[1860,1855],[2355,1850],[2950,1850],[3425,1820],[3920,1830],[4160,1775],[4650,1765],[5150,1775]]
    width,height = 1600,1000
    
    # Read Img
    wf = cv2.imread("./wf.JPG")
    
    files = os.listdir("./")
    #os.chdir("./")
    for img_name in glob.glob("*.JPG"):

        print("img_name: ",img_name)
        best_score = -999
        best_area_index = -999

        if img_name[:-4] != "wf":
            mf = cv2.imread(img_name)
            for i,area in enumerate(search_areas):
                print("Area ",i," - ", end = '')

                img1, img2 = preprocessImg(wf,mf,area,width,height)
                score = matchORB(img1,img2,False)
                #score = matchSIFT(img1,img2,False)
                if score>best_score:
                    best_score = score
                    best_area = area

            # Visualize The Best Fit Area
            img1, img2 = preprocessImg(wf,mf,best_area,width,height)
            score = matchORB(img1,img2,True)
            #score = matchSIFT(img1,img2,True)
            cv2.waitKey(0)


