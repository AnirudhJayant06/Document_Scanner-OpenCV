import cv2
import numpy as np

# Defining width & height of the video
video_width = 640
video_height = 480

cap = cv2.VideoCapture(1)

cap.set(3, video_width)
cap.set(4, video_height)

# Setting brightness of the video
cap.set(10, 1)


def getEdges(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 1.2)
    img_canny = cv2.Canny(img_blur, 100, 100)
    kernel = np.ones((5,5))
    img_dila = cv2.dilate(img_canny, kernel, iterations=5)
    img_ero = cv2.erode(img_dila, kernel, iterations=2)

    return img_ero


# First we find the biggest contour &, then, do the looping on that
def getContours(func_img):
    max_area = 0
    biggest = np.array([])
    # Our contours got saved in 'contours'
    contours, hierarchy = cv2.findContours(func_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # To look into each individual contour
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # Drawing contours
            # Negative value (i.e. -1) means draw all contours
            #cv2.drawContours(img_conto, cnt, -1, (255, 0, 0), 3)

            # Calculating the perimeters of all contours
            peri = cv2.arcLength(cnt, True)
            # Calculating the approximate no. of points of each contour
            approx_points = cv2.approxPolyDP(cnt, 0.01 * peri, True)

            # Finding biggest contour i.e the biggest rectangle/ square box coordinates
            # or the outermost edges coordinates
            if area > max_area and len(approx_points) == 4:
                biggest = approx_points
                max_area = area

    #Drawing the biggest contours on 'img_conto' image
    cv2.drawContours(img_conto, biggest, -1, (255, 0, 0), 20)
    return biggest

def reorder(points):
    '''
    Since it's previous shape was 4x1x2 where '1' was redundant and will create an issue while adding the
     coordinates.
    '''

    points = points.reshape((4,2))

    #Creating a new array of zeros to insert the rearranged values
    reordered_points = np.zeros((4,1,2),np.int32)

    # This calculates the sum of each points and store it in an array. Here '1' indicate the axis along which
    # summation needs to be done.
    sum = points.sum(1)

    # This searches for the index of the minimum value in the 'sum' array and then assign that point
    # to the first index (i.e. 0) in the reordered array
    reordered_points[0] = points[np.argmin(sum)]

    # This searches for the index of the maximum value in the 'sum' array and then assign that point
    # to the last index (i.e. 3) in the reordered array
    reordered_points[3] = points[np.argmax(sum)]

    '''
    When you 'll substract width - 0, it 'll give the highest positive values and when you 'll subtract
    0 - height, it 'll give the highest negative value because rest of the points are (0,0) & (Width, Height)
    which can neither give the highest value or the lowest value on subtraction
    '''
    # Calculating difference of the points
    diff = np.diff(points,axis=1)

    # Arranging points as per our requirement
    reordered_points[1] = points[np.argmin(diff)]
    reordered_points[2] = points[np.argmax(diff)]

    return reordered_points


def getWarp(img, biggest):

    biggest = reorder(biggest)
    #print("Biggest ",biggest)
    # Assigning an array of 4 points/ coordinates to the first point
    pt1 = np.float32(biggest)
    '''
    This mapping of points, first point in the 'biggest' array 'll be (0,0) i.e. minimum
     & last point 'll be (width, height) i.e. maximum, is not correct. We need to reorder these points.
     If we analyze, we can notice that sum of x & y corrdinates 'll be minimum for (0,0) & maximum for 
     (width, height). So, using this, we 'll write a function for reordering.
    '''
    # Defining the coordinates as per our requirement of the new image
    pt2 = np.float32([[0, 0], [video_width, 0], [0, video_height], [video_width, video_height]])

    # Creating Perspective Transform Filter using pt1 & pt2. This 'll transform (rotate) the image.
    matrix = cv2.getPerspectiveTransform(pt1, pt2)

    # Getting Warp Perspective of the required image
    output_img = cv2.warpPerspective(img, matrix, (video_width, video_height))

    #Shredding the extra edges/ noise around the warpped image
    cropped_img = output_img[5:output_img.shape[0] - 5, 5:output_img.shape[1] - 5]
    #Resizing the cropped image
    cropped_img = cv2.resize(cropped_img, (video_width, video_height))

    return cropped_img

def Combine2Images(rgb_image, gray_image):

    rgb_rows, rgb_cols, channels = rgb_image.shape
    gray_rows, gray_cols = gray_image.shape

    # Choosing the maximum no. of rows so that no image gets cropped
    comb_rows = max(rgb_rows, gray_rows)

    # As we need to place 2 images side by side, so our final image width should be the sum of columns of both the image
    comb_cols = rgb_cols + gray_cols

    # Creating a new array of zeros as per our new dimensions to set both the images side by side
    final_img = np.zeros(shape=(comb_rows,comb_cols,channels),dtype=np.uint8)

    # Setting images side by side
    final_img[:rgb_rows,:rgb_cols] = rgb_image

    # Setting images side by side
    # Since, it's a gray image so it 'll have only 2 channels. Therefore, we will add 'None' value in the 3rd channel
    final_img[:gray_rows,rgb_cols:] = gray_image[:,:,None]

    return final_img


def rescaleImg(img, scale):
    new_width = int(img.shape[1] * (scale / 100))
    new_height = int(img.shape[0] * (scale / 100))
    dim = (new_width, new_height)

    #Down Scaling an Image
    if scale <= 100:
        inter = cv2.INTER_AREA
    # Up Scaling an Image
    elif scale > 100:
        inter = cv2.INTER_CUBIC

    # Resizing an Image
    resized_img = cv2.resize(img, dim, interpolation=inter)

    return resized_img


while True:
    # This loads each frame into an img
    Success, img = cap.read()

    # Resizing the image
    img = cv2.resize(img, (video_width, video_height))

    # Copying an image to draw contours on it
    img_conto = img.copy()

    # To get edges of an image
    edges_img = getEdges(img)

    # This returns the biggest contour i.e. outermost boundary coordinates
    biggest_contour = getContours(edges_img)

    '''
    If there is no document to scan, then, 'biggest_contour' 'll not have have anything i.e. it's shape 'll be
    (0,0,0) and not (4,2,1). Since, In the very next step we call 'getWarp' function which 'll call 'reorder'
    function that 'll try to reshape this (biggest_contour) array to (4,2) which 'll throw an error if the 
    array is not in (x, y, z) format.
    
    Therefore, to avoid that, we 'll check & proceed only if its (4,2,1) otherwise we 'll assign original image
    to warp image    
    '''
    if biggest_contour.size != 0:

        #print("Biggest: ",biggest_contour)
        Warp_img = getWarp(img,biggest_contour)
    else:
        Warp_img = img

    # This sets the image size to 60% i.e if your image is 100x100 px, then, it 'll be 60x60 px
    scale=60

    #Downscaling all images
    resized_img = rescaleImg(img, scale)
    resized_Thresh_img = rescaleImg(edges_img, scale)
    resized_img_conto = rescaleImg(img_conto, scale)
    resized_Warp_img = rescaleImg(Warp_img, scale)

    # Combining/ Stacking original image (rgb scale) with image having edges (gray scale) horizontally
    final_img_1 = Combine2Images(resized_img,resized_Thresh_img)

    # Combining/ Stacking contoured image & warp image horizontally
    final_img_2 = np.hstack((resized_img_conto,resized_Warp_img))

    # Verically stacking the above to combined images
    final_img = np.vstack((final_img_1,final_img_2))

    # Inseting text on video
    cv2.putText(final_img, "Press S to stop", ((video_width // 2) - 100, 30), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 0, 255), 2)

    cv2.imshow("Result", final_img)
    # This defines the waiting time & sets exit key to 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
