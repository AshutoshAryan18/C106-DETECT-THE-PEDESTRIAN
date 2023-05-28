import cv2


# Create our body classifier
vid = cv2.VideoCapture(r"C:\Users\ASHUTOSH ARYAN\OneDrive\Desktop\coding\module 3\project\C---106[DETECT THE PEDESTRIAN]\PRO-106-ProjectTemplate-main\walking.avi")
full_Body_cascade=cv2.CascadeClassifier(r"C:\Users\ASHUTOSH ARYAN\OneDrive\Desktop\coding\module 3\project\C---106[DETECT THE PEDESTRIAN]\PRO-106-ProjectTemplate-main\haarcascade_fullbody.xml")

while True:
    ret, frame = vid.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodys = full_Body_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in bodys:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',frame)
        # Quit Window by Spacebar Key
    if cv2.waitKey(25) == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()

