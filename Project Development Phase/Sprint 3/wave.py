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

fp_wave_test_healthy = './TestData/wave/healthy'
fp_wave_test_park = './TestData/wave/parkinson'

wave_test_healthy = os.listdir(fp_wave_test_healthy)
wave_test_park = os.listdir(fp_wave_test_park)

testXW = []
testYW = []

for i in wave_test_healthy:
  img = cv2.imread(fp_wave_test_healthy + '/' + i)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (200,200))
  img = cv2.threshold(img, 0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  features = quantify(img)
  testXW.append(features)
  testYW.append('healthy')

for i in wave_test_park:
  img = cv2.imread(fp_wave_test_park + '/' + i)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (200,200))
  img = cv2.threshold(img , 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  features = quantify(img)
  testXW.append(features)
  testYW.append('parkinson')

testYW = le.fit_transform(testYW)

with open('./models/wave.pkl' , 'rb') as f:
    model = pickle.load(f)

predW = model.predict(testXW)

print("\nWAVE DRAWINGS:-")
print("Model - K-Neighbor Classifier\n")

print("Healthy --> 0    Parkinson --> 1")
print("Prediction: ", predW)
print("Actual: ", testYW)

print("\n\nMODEL EVALUATION:-\n")

acc = accuracy_score(testYW, predW)
print("Accuracy: ", acc)

rec = recall_score(testYW, predW)
print("Recall: ", rec)

pre = precision_score(testYW, predW)
print("Precision: ", pre)

f1 = f1_score(testYW, predW)
print("F1 score: ", f1)

cnf = confusion_matrix(testYW, predW)
print("Confusion Matrix: \n",cnf)

plt.figure(figsize = (7,7))
sns.heatmap(cnf , annot = True , cmap = "tab10" , cbar=False)
plt.show()