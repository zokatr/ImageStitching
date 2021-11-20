import cv2
import time
import glob, os
import numpy as np
from cv2.xfeatures2d import matchGMS



def preprocessImg(wf,mf,area,width,height,resize_scale):
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
    w = int(width/resize_scale)
    h = int(height/resize_scale)
    dim = (w, h)
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(mf, dim, interpolation = cv2.INTER_AREA)
    print("\nresize_sclae: ",resize_scale,end = '')
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


def visualizePosition(img1,img2,kp1,kp2,matches,title,ransac_th):
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
    try:
        H, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, ransac_th)
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
    save_img_name = "./saved_img/img-"+str(resize_scale)+"-"+str(ransac_th)+"-"+str(orb_count)+"-"+str(gms_th)+".png"
    cv2.imwrite(save_img_name,output)





def matchORB(img1,img2,last_call,gms_th,orb_count,ransac_th):
    """
    Detection & matching the features with ORB
    img1 : wide field
    img2 : medium field
    last_call: visualization flag - True/False
    """
    
    # ORB Feature Detect & Match
    orb = cv2.ORB_create(10000*orb_count)
    orb.setFastThreshold(0)
    kp1, des1 = orb.detectAndCompute(img1, None)    
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = bf.match(des1, des2)

    # Filter the matches
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=True, withRotation=False, thresholdFactor=gms_th)
    
    # Draw Outputs
    if last_call:
        visualizePosition(img1,img2,kp1,kp2,matches_gms,"ORB",ransac_th)



    img_diff = calculateDifference(img1,img2)
    print(" ransac_th: ",ransac_th,"  gms_th: ",gms_th," orb_count: ",orb_count)
    print("ORB-GMS Match count : ",len(matches_gms)," Diff: ",img_diff)

    # To detect the matching area in wide field img, return the count of feature match
    score = len(matches_gms)
    return score

def updateResize(resize_scale):
    #pass
    resize_scale = cv2.getTrackbarPos('resize','control_window')
    ransac_th    = cv2.getTrackbarPos('ransac','control_window')
    orb_count    = cv2.getTrackbarPos('orb','control_window')
    gms_th    = cv2.getTrackbarPos('gms','control_window')

    if resize_scale == 0: resize_scale = 1
    img1, img2 = preprocessImg(wf,mf,best_area,width,height,resize_scale)
    score = matchORB(img1,img2,True,gms_th,orb_count,ransac_th)
def updateRansac(ransac_th):
    #pass
    resize_scale = cv2.getTrackbarPos('resize','control_window')
    ransac_th    = cv2.getTrackbarPos('ransac','control_window')
    orb_count    = cv2.getTrackbarPos('orb','control_window')
    gms_th    = cv2.getTrackbarPos('gms','control_window')
    img1, img2 = preprocessImg(wf,mf,best_area,width,height,resize_scale)
    score = matchORB(img1,img2,True,gms_th,orb_count,ransac_th)
def updateORB(orb_count):
    #pass
    resize_scale = cv2.getTrackbarPos('resize','control_window')
    ransac_th    = cv2.getTrackbarPos('ransac','control_window')
    orb_count    = cv2.getTrackbarPos('orb','control_window')
    gms_th    = cv2.getTrackbarPos('gms','control_window')
    img1, img2 = preprocessImg(wf,mf,best_area,width,height,resize_scale)
    score = matchORB(img1,img2,True,gms_th,orb_count,ransac_th)
def updateGMS(gms_thres):
    #pass
    resize_scale = cv2.getTrackbarPos('resize','control_window')
    ransac_th    = cv2.getTrackbarPos('ransac','control_window')
    orb_count    = cv2.getTrackbarPos('orb','control_window')
    gms_th    = cv2.getTrackbarPos('gms','control_window')
    img1, img2 = preprocessImg(wf,mf,best_area,width,height,resize_scale)
    score = matchORB(img1,img2,True,gms_th,orb_count,ransac_th)
def updateSave(x):
    #pass
    img1, img2 = preprocessImg(wf,mf,best_area,width,height,resize_scale)
    score = matchORB(img1,img2,True,gms_th,orb_count,ransac_th)
    save_img_name = "./saved_img/img-",resize_scale,"-",ransac_th,"-",orb_count,"-",gms_th,".png"
    #cv2.imwrite(save_img_name,output)





#start = time.time()
search_areas = [[95,1945],[510,1990],[1015,1855],[1500,1700],[1860,1855],[2355,1850],[2950,1850],[3425,1820],[3920,1830],[4160,1775],[4650,1765],[5150,1775]]
width,height = 1600,1000

# Read Img
wf = cv2.imread("./wf.JPG")

files = os.listdir("./")
img_list = []
img2_index = 0
best_area = [0,0]
best_score = -999
best_area_index = -999
for img_name in glob.glob("*.JPG"):
    img_list.append(img_name)




cv2.namedWindow('control_window')
cv2.resizeWindow('control_window',800,200)
# create trackbars for color change
cv2.createTrackbar('resize','control_window',1,10,updateResize)
cv2.createTrackbar('ransac','control_window',1,10,updateRansac)
cv2.createTrackbar('orb','control_window',1,3,updateORB)
cv2.createTrackbar('gms','control_window',1,10,updateGMS)
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'control_window',0,1,updateSave)


cv2.setTrackbarPos('resize','control_window',2)
cv2.setTrackbarPos('ransac','control_window',6)
cv2.setTrackbarPos('orb','control_window',1)
cv2.setTrackbarPos('gms','control_window',6)
resize_scale = cv2.getTrackbarPos('resize','control_window')
ransac_th    = cv2.getTrackbarPos('ransac','control_window')
orb_count    = cv2.getTrackbarPos('orb','control_window')
gms_th    = cv2.getTrackbarPos('gms','control_window')

while(1):
    best_area = [0,0]
    best_score = -999
    best_area_index = -999
    img_name = img_list[img2_index]
    if img_name[:-4] != "wf":
        mf = cv2.imread(img_name)
    

    for i,area in enumerate(search_areas):
        print("Area ",i," - ", end = '')
        
        if i>1:
            continue
        area = search_areas[int(img_name[2:-4])]
        img1, img2 = preprocessImg(wf,mf,area,width,height,resize_scale)
        score = matchORB(img1,img2,False,gms_th,orb_count,ransac_th)

        if score>best_score:
            best_score = score
            best_area = area
            best_area_index = i


    # Visualize The Best Fit Area
    img1, img2 = preprocessImg(wf,mf,best_area,width,height,resize_scale)
    score = matchORB(img1,img2,True,gms_th,orb_count,ransac_th)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    if k == ord('d'):
        img2_index += 1



"""
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

            if i>1:
                continue
            
            area = search_areas[int(img_name[2:-4])]
            img1, img2 = preprocessImg(wf,mf,area,width,height)
            score = matchORB(img1,img2,False)

            if score>best_score:
                best_score = score
                best_area = area
                best_area_index = i
            if i>1:
                continue

        # Visualize The Best Fit Area
        img1, img2 = preprocessImg(wf,mf,best_area,width,height)
        score = matchORB(img1,img2,True)
        k = cv2.waitKey(0)
        if k == ord('q'):
            cv2.destroyAllWindows()


"""
