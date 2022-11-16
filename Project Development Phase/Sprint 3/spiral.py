# Importing necessary libraries
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score

# Quantifying Images
def quantify(img):
  features = feature.hog(img, orientations = 9, pixels_per_cell = (10,10), cells_per_block = (2 , 2), transform_sqrt = True, block_norm = "L1")
  return features


le = LabelEncoder()     # Label Encoder

# Specifying Image Path
fp_spiral_test_healthy = './TestData/spiral/healthy'
fp_spiral_test_park = './TestData/spiral/parkinson'

spiral_test_healthy = os.listdir(fp_spiral_test_healthy)
spiral_test_park = os.listdir(fp_spiral_test_park)

# Spliting into dependant and independant data
testX = []
testY = []

for i in spiral_test_healthy:
  img = cv2.imread(fp_spiral_test_healthy + '/' + i)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (200,200))
  img = cv2.threshold(img, 0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  features = quantify(img)
  testX.append(features)
  testY.append('healthy')

for i in spiral_test_park:
  img = cv2.imread(fp_spiral_test_park + '/' + i)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (200,200))
  img = cv2.threshold(img , 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  features = quantify(img)
  testX.append(features)
  testY.append('parkinson')

testY = le.fit_transform(testY) # Fit and Transforming / Encoding

# Loading Pre-Trained Pickle Model
with open('./models/spiral.pkl' , 'rb') as f:
    model = pickle.load(f)

pred = model.predict(testX) # Predicting using model

print("\nSPIRAL DRAWINGS:-")
print("Model - Random Forest Classifier\n")

print("Healthy --> 0    Parkinson --> 1")
print("Prediction: ", pred)
print("Actual: ", testY)

print("\n\nMODEL EVALUATION:-\n")

# Performing model evaluation using various metrics by comparing prediction and actual values

acc = accuracy_score(testY, pred)
print("Accuracy: ", acc)

rec = recall_score(testY, pred)
print("Recall: ", rec)

pre = precision_score(testY, pred)
print("Precision: ", pre)

f1 = f1_score(testY, pred)
print("F1 score: ", f1)

cnf = confusion_matrix(testY, pred)
print("Confusion Matrix: \n",cnf)

plt.figure(figsize = (7,7))
sns.heatmap(cnf , annot = True , cmap = "tab10" , cbar=False)
plt.show()