import numpy as np
import cv2
import json

img = cv2.imread('0.png')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(imgray, 128, 255, 0)

# Detect contours using both methods on the same image
(_, cnts, _)= cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

final =[]
for pick in cnts :
    final.append([[ int( u[0][0]), int(u[0][1])] for u in pick  ])

# print( final )
 
ret = [] 
counter =0
for pick in final:
    counter+=1 
    ret.append( { "coordinates": pick   } )


json_dump = json.dumps( ret )

with open( 'regions.json','w' ) as file:
    file.write(  json_dump )

# Draw both contours onto the separate images
cv2.drawContours(img, cnts, -1, (255,0,0), 2)


# Now show the image
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()