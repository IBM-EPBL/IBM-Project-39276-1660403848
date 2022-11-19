# Importing necessary libraries
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, PrecisionRecallDisplay, RocCurveDisplay

# Quantifying Images to get features
def get_features(img):
  features = feature.hog(img, orientations = 9, pixels_per_cell = (10,10), cells_per_block = (2 , 2), transform_sqrt = True, block_norm = "L1")
  return features

# Preprocessing Images
def preprocess_images(img, data):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200))
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = get_features(img)
    data.append(features)
    return data

le = LabelEncoder()     # Label Encoder

# Specifying Image Path
fp_wave_test_healthy = './TestData/wave/healthy'
fp_wave_test_park = './TestData/wave/parkinson'

wave_test_healthy = os.listdir(fp_wave_test_healthy)
wave_test_park = os.listdir(fp_wave_test_park)

# Spliting into dependant and independant data
testX = []
testY = []

for i in wave_test_healthy:
    image = cv2.imread(fp_wave_test_healthy + '/' + i)
    testX = preprocess_images(image, testX)
    testY.append('healthy')

for i in wave_test_park:
    image = cv2.imread(fp_wave_test_park + '/' + i)
    testX = preprocess_images(image, testX)
    testY.append('parkinson')

testY = le.fit_transform(testY)   # Fit and Transforming / Encoding

# Loading Pre-Trained Pickle Model
with open('./models/wave.pkl' , 'rb') as f:
    model = pickle.load(f)

pred = model.predict(testX)   # Predicting using model

print("\nWAVE DRAWINGS:-")
print("Model - K-Neighbor Classifier\n")

print("Healthy --> 0    Parkinson --> 1")
print("Prediction: ", pred)
print("Actual: ", testY)


print("\n\nMODEL EVALUATION:-\n")

# Performing model evaluation using various metrics by comparing prediction and actual values

acc = accuracy_score(testY, pred)
print("Accuracy: ", acc)

rec = recall_score(testY, pred)
print("Recall: ", rec)

spe = recall_score(testY, pred, pos_label = 0)
print("Specificity: ", spe)

pre = precision_score(testY, pred)
print("Precision: ", pre)

f1 = f1_score(testY, pred)
print("F1 score: ", f1)

cnf = confusion_matrix(testY, pred)
print("Confusion Matrix: \n",cnf)

fpr,tpr, thresholds = roc_curve(testY, pred)
print("ROC CURVE:- \nFpr: {} \nTpr: {} \nThreshold: {}".format(fpr,tpr,thresholds))

p, r, t = precision_recall_curve(testY, pred)
print("PR CURVE:- \nPrecision: {} \nRecall: {} \nThreshold: {}".format(p,r,t))

roc = roc_auc_score(testY, pred)
print("ROC AUC Score: ",roc)

RocCurveDisplay.from_predictions(testY, pred)

PrecisionRecallDisplay.from_predictions(testY, pred)

plt.figure(figsize = (7,7))
sns.heatmap(cnf , annot = True , cmap = "tab10" , cbar=False)
plt.show()
