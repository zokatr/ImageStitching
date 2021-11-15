import numpy as np
from enum import Enum
import time
import cv2
from cv2.xfeatures2d import matchGMS
import matplotlib.pyplot as plt


class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5


def alignImg(im1,im2):
        # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

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
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned

def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None

if __name__ == '__main__':
    
    # Read Img
    wf = cv2.imread("./wf.jpg")
    mf = cv2.imread("./mf00.jpg")
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
    
    # Feature Detect & Match
    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)

    descriptor = cv2.xfeatures2d.SIFT_create()


    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING) # NORM_HAMMING
    matches_all = matcher.match(des1, des2)

    start = time.time()
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=True, withRotation=False, thresholdFactor=6)
    end = time.time()

    print('Found', len(matches_gms), 'matches')
    print('GMS takes', end-start, 'seconds')
    

    #-------------------------------------------------------------------------------
    """
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches_gms]
    list_kp2 = [kp2[mat.queryIdx].pt for mat in matches_gms]

    print(list_kp1[0][1])
    print(list_kp2[0][1])

    M = getHomography(kp1, kp2, des1, des2, matches_gms, reprojThresh=50)
    if M is None:
        print("Error!")
    (matches, H, status) = M
    print(H)


    img_matches = img1#np.empty((max(img2.shape[0], img1.shape[0]), img2.shape[1]+img1.shape[1], 3), dtype=np.uint8)
    #-- Get the corners from the image_1 ( the object to be "detected" )
    img_object = img2

    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0+3
    obj_corners[0,0,1] = 0+3
    obj_corners[1,0,0] = img_object.shape[1]-3
    obj_corners[1,0,1] = 0+3
    obj_corners[2,0,0] = img_object.shape[1]-3
    obj_corners[2,0,1] = img_object.shape[0]-3
    obj_corners[3,0,0] = 0+3
    obj_corners[3,0,1] = img_object.shape[0]-3

    dst  = cv2.perspectiveTransform(obj_corners, H)
    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    img_matches = cv2.polylines(img_matches,[np.int32(dst)],True,(255,0,0),20, cv2.LINE_AA)
    cv2.imshow('Good Matches & Object detection', img_matches)
    print(img_matches.shape)




    
    trainImg = img1
    queryImg = img2
    width = trainImg.shape[1] #+ queryImg.shape[1]
    height = trainImg.shape[0] #+ queryImg.shape[0]

    result = cv2.warpPerspective(img_object, H, (width, height))
    #result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

    print(result.shape)
    cv2.imshow("r",result)
    """

    #-------------------------------------------------------------------------------
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches_gms ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches_gms ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #pts = np.float32([ [0,0],[0,150],[150,150],[151,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    print(pts)
    rect = cv2.polylines(img1,[np.int64(dst)],True,(255,0,255),3, cv2.LINE_AA)
    #dim = (width*2, 2*height)
    #rect = cv2.resize(rect, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("rectangle",rect)
    #-------------------------------------------------------------------------------
    # Draw Outputs
    output = draw_matches(img1, img2, kp1, kp2, matches_gms, DrawingType.ONLY_LINES)
    


    cv2.imshow("show", output)
    cv2.waitKey(0)
