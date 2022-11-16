import os
import cv2
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score

def quantify(img):
  features = feature.hog(img, orientations = 9, pixels_per_cell = (10,10), cells_per_block = (2 , 2), transform_sqrt = True, block_norm = "L1")
  return features

le = LabelEncoder()

fp_spiral_test_healthy = './TestData/spiral/healthy'
fp_spiral_test_park = './TestData/spiral/parkinson'

spiral_test_healthy = os.listdir(fp_spiral_test_healthy)
spiral_test_park = os.listdir(fp_spiral_test_park)

testXS = []
testYS = []

for i in spiral_test_healthy:
  img = cv2.imread(fp_spiral_test_healthy + '/' + i)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (200,200))
  img = cv2.threshold(img, 0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  features = quantify(img)
  testXS.append(features)
  testYS.append('healthy')

for i in spiral_test_park:
  img = cv2.imread(fp_spiral_test_park + '/' + i)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (200,200))
  img = cv2.threshold(img , 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  features = quantify(img)
  testXS.append(features)
  testYS.append('parkinson')

testYS = le.fit_transform(testYS)

with open('./models/spiral.pkl' , 'rb') as f:
    model = pickle.load(f)

predS = model.predict(testXS)

print("\nSPIRAL DRAWINGS:-")
print("Model - Random Forest Classifier\n")

print("Healthy --> 0    Parkinson --> 1")
print("Prediction: ", predS)
print("Actual: ", testYS)

print("\n\nMODEL EVALUATION:-\n")

acc = accuracy_score(testYS, predS)
print("Accuracy: ", acc)

rec = recall_score(testYS, predS)
print("Recall: ", rec)

pre = precision_score(testYS, predS)
print("Precision: ", pre)

f1 = f1_score(testYS, predS)
print("F1 score: ", f1)

cnf = confusion_matrix(testYS, predS)
print("Confusion Matrix: \n",cnf)

plt.figure(figsize = (7,7))
sns.heatmap(cnf , annot = True , cmap = "tab10" , cbar=False)
plt.show()