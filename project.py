import cv2
import matplotlib.pyplot as plt
import numpy as np

print("Artificial Intelligence Project: Compare 2 Images by Munim, Shahmeer and Ayesha")

image1=cv2.imread("E:/uni/ai/Project/train.jpg")
training_image=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
training_gray=cv2.cvtColor(training_image,cv2.COLOR_RGB2GRAY)

image2=cv2.imread("E:/uni/ai/Project/test.jpg")
test_image=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
test_image=cv2.pyrDown(test_image)

num_rows,num_cols=test_image.shape[:2]
rotation_matrix=cv2.getRotationMatrix2D((num_cols/2,num_rows/2),30,1)
test_image=cv2.warpAffine(test_image,rotation_matrix,(num_cols,num_rows))
test_gray=cv2.cvtColor(test_image,cv2.COLOR_RGB2GRAY)

fx,plots=plt.subplots(1,2,figsize=(20,10))
plots[0].set_title("Training Image")
plots[0].imshow(training_image)
plots[1].set_title("Testing Image")
plots[1].imshow(test_image)
orb=cv2.ORB_create()

train_keypoints,train_descriptor=orb.detectAndCompute(training_gray,None)
test_keypoints,test_descriptor=orb.detectAndCompute(test_gray,None)
keypoints_without_size=np.copy(training_image)
keypoints_with_size=np.copy(training_image)

cv2.drawKeypoints(training_image,train_keypoints,keypoints_without_size,color=(0,255,0))
cv2.drawKeypoints(training_image,train_keypoints,keypoints_with_size,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fx,plots=plt.subplots(1,2,figsize=(20,10))

plots[0].set_title("Training keypoints with size")
plots[0].imshow(keypoints_with_size,cmap='gray')
plots[1].set_title("Training keypoints without size")
plots[1].imshow(keypoints_without_size,cmap='gray')

print("\nNumber of keypoints detected in the training image: ",len(train_keypoints))
print("Number of keypoints detected in the testing image: ",len(test_keypoints))
ratio=(len(test_keypoints)/len(train_keypoints))*100

bf=cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
matches=bf.match(train_descriptor,test_descriptor)
matches=sorted(matches,key=lambda x :x.distance)
result=cv2.drawMatches(training_image,train_keypoints,test_gray,test_keypoints,matches,test_gray,flags=2)

plt.rcParams['figure.figsize']=[14.0,7.0]
plt.title('Best matching points')
plt.imshow(result)
plt.show()

print("\nTraining and Testing Image Matching Ratio: ",ratio,"%")