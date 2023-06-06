# Document_Scanner-OpenCV
Using OpenCV library, I access the externally attached webcam and scan the document. It detects the 4 corners, crops the background &amp; rotate the picture so that edges are always parallel to horizontal &amp; vertical axis.

Steps:

1. Capture the frame containing the document using external webcam.
2. 'getEdges' function is called (which uses gaussian blur, canny, dialation and errosion to get all the edges) which returns an image with edges.
3. This image (image with edges) is passed onto an another function called 'getContours' which draws & return the biggest contour. 
4. 'getWarp' function is called which calls another function 'reorder' which 'll arrange all the four points in the right order and returns reordered points. 'getWarp' function 'll use the reordered points to crop the background from the image and rotate the image as per our requirement. It 'll return this cropped & rotated image.
5. 'rescaleImg' function is called to rescale the images so that we can place them side-by-side. 
6. Finally, 'combine2Images' function is called to combine all the 4 images and displayed as an output.

Project Demo: https://drive.google.com/file/d/1482qkEXQWHyGDW4eI1lyT2UgHmRmlw0g/view?usp=sharing
