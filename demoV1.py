"""
11.17.21
Ahmet Gazi ÇİFCİ
"""
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

def calculateDifference(img1,img2):


    #im1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #im2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    s1 = cv2.mean(img1)
    s2 = cv2.mean(img2)
    #g1 = cv2.mean(im1_gray)
    #g2 = cv2.mean(im2_gray)

    r = abs(s1[0]-s2[0])
    g = abs(s1[1]-s2[1])
    b = abs(s1[2]-s2[2])
    color_sum = r+g+b
    return color_sum

def detectFeature(img):

    # ORB Feature Detect
    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)
    kp1, des1 = orb.detectAndCompute(img, None)
    
    return [kp1,des1]


def visualizePosition(img1,img2,kp1,kp2,matches,title):
    """
    Draw a rectangle to matched area &  Overlay the medium image to wide image
    img1 : wide field
    img2 : medium field
    title: algorithm name
    """
    # -------------------------
    """
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4,2))
    kernel = np.ones((11, 11), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)


    im1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    th1=50
    th2=100
    img1_e = cv2.Canny(im1_gray,th1,th2,21,L2gradient=True)
    img2_e = cv2.Canny(im2_gray,th1,th2,21,L2gradient=True)
    
    
    img1_e = cv2.adaptiveThreshold(im1_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 51, 2)
    img2_e = cv2.adaptiveThreshold(im2_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 51, 2)
    

    img1_e = cv2.erode(img1_e, kernel2)
    img2_e = cv2.erode(img2_e, kernel2)
    img1_e = cv2.dilate(img1_e, kernel)
    img2_e = cv2.dilate(img2_e, kernel)
    
    
    cv2.imshow("1-e",img1_e)
    cv2.imshow("2-e",img2_e)
    """
    # -------------------------

    # Find the homography matrix
    h,w,d = img1.shape
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    try:
        H, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, 6.0)
    except:
        return 0
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
    #cv2.imshow(title,overlayed_img)
    cv2.imshow("matches",output)





def matchORB(img1,img2,feature,last_call):
    """
    Detection & matching the features with ORB
    img1 : wide field
    img2 : medium field
    last_call: visualization flag - True/False
    """
    
    # ORB Feature Detect & Match
    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)
    start = time.time()
    kp1, des1 = feature[0],feature[1]#orb.detectAndCompute(img1, None)    
    kp2, des2 = orb.detectAndCompute(img2, None)
    end = time.time()
    print("TIME : ", end-start)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = bf.match(des1, des2)

    # Filter the matches
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=True, withRotation=False, thresholdFactor=6)
    
    # Draw Outputs
    if last_call:
        visualizePosition(img1,img2,kp1,kp2,matches_gms,"ORB")


    img_diff = calculateDifference(img1,img2)

    print("ORB-GMS Match count : ",len(matches_gms)," Diff: ",img_diff)

    # To detect the matching area in wide field img, return the count of feature match
    score = len(matches_gms)
    return score

def matchECC(im1,im2,last_call):

    # Find size of image1
    sz = im1.shape
    h,w,d = im1.shape

    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)


    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;
    
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=15)
    

    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)     
    dst = cv2.perspectiveTransform(pts,warp_matrix)
    img_rectangle = cv2.polylines(im1,[np.int32(dst)],True,(255,255,255),1, cv2.LINE_AA)
    cv2.imshow("rectanglea",img_rectangle)


    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    # Show final results
    cv2.imshow("Image 1", im1)
    cv2.imshow("Image 2", im2)
    cv2.imshow("Aligned Image 2", im2_aligned)
    cv2.waitKey(0)

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

def akaze_match(img1,img2,last_call):
    # load the image and convert it to grayscale
  

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(img1, None)
    kpts2, desc2 = akaze.detectAndCompute(img2, None)



    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    try:
        nn_matches = matcher.knnMatch(desc1, desc2, 2)
    except:
        return 0

    matched1 = []
    matched2 = []
    nn_match_ratio = 0.1 # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])

    print ("asdasdasdasd" ,len(nn_matches))
    h,w,d = img1.shape
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in nn_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in nn_matches ]).reshape(-1,1,2)
    homography, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, 6.0)


    inliers1 = []
    inliers2 = []
    good_matches = []
    inlier_threshold = 2.5 # Distance threshold to identify inliers with homography check
    for i, m in enumerate(matched1):
        col = np.ones((3,1), dtype=np.float64)
        col[0:2,0] = m.pt
        col = np.dot(homography, col)
        col /= col[2,0]
        dist = sqrt(pow(col[0,0] - matched2[i].pt[0], 2) +\
                    pow(col[1,0] - matched2[i].pt[1], 2))
        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])

    res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, inliers1, img2, inliers2, good_matches, res)
    print('A-KAZE Matching Results')
    print('*******************************')
    print('# Keypoints 1:                        \t', len(kpts1))
    print('# Keypoints 2:                        \t', len(kpts2))
    print('# Matches:                            \t', len(matched1))
    print('# Inliers:                            \t', len(inliers1))
    cv2.imshow('result', res)
    cv2.waitKey(0)
    """
    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, good[1:20], None, flags=2)
    cv2.imshow("AKAZE matching", img3)
    cv2.waitKey(0) """

if __name__ == '__main__':

    #start = time.time()
    search_areas = [[95,1945],[510,1990],[1015,1855],[1500,1700],[1860,1855],[2355,1850],[2950,1850],[3425,1820],[3920,1830],[4160,1775],[4650,1765],[5150,1775]]
    width,height = 1600,1000
    featureList = []
    # Read Img
    wf = cv2.imread("./wf.JPG")
    
    for i,area in enumerate(search_areas):
        
        img_crop, _ = preprocessImg(wf,wf,area,width,height)
        kp_des = detectFeature(img_crop)
        featureList.append(kp_des)


    files = os.listdir("./")
    #os.chdir("./")
    for img_name in glob.glob("*.JPG"):

        print("img_name: ",img_name)
        best_score = -999
        best_area_index = -999

        #if img_name[2:-4] != "02":
        #    continue

        if img_name[:-4] != "wf":
            mf = cv2.imread(img_name)

            for i,area in enumerate(search_areas):
                print("Area ",i," - ", end = '')


                #area = search_areas[2]
                img1, img2 = preprocessImg(wf,mf,area,width,height)
                score = matchORB(img1,img2,featureList[i],False)
                #score = matchSIFT(img1,img2,False)
                #matchECC(img1,img2,False)
                #akaze_match(img1,img2,False)
                #score = 0
                if score>best_score:
                    best_score = score
                    best_area = area
                    best_area_index = i

            # Visualize The Best Fit Area
            img1, img2 = preprocessImg(wf,mf,best_area,width,height)
            score = matchORB(img1,img2,featureList[best_area_index],True)
            #score = matchSIFT(img1,img2,True)
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()




