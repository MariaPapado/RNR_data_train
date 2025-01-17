from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#def extract_statistical_features(signal):
#    return {
#        "mean": np.mean(signal),
#        "variance": np.var(signal),
#        "skewness": np.mean((signal - np.mean(signal)) ** 3) / (np.std(signal) ** 3),
#        "kurtosis": np.mean((signal - np.mean(signal)) ** 4) / (np.std(signal) ** 4),
#        "min": np.min(signal),
#        "max": np.max(signal)
#    }


# Example feature extraction functions
def extract_histogram_features(pixels, bins=10):
    hist, _ = np.histogram(pixels, bins=bins, range=(np.min(pixels), np.max(pixels)))
    hist_normalized = hist / np.sum(hist)
    return hist_normalized  # Use normalized histogram

def extract_statistical_features(pixels):
    #pixels = pixels/255.
    mean = np.mean(pixels)
    variance = np.var(pixels)
    skewness = np.mean((pixels - mean) ** 3) / (np.std(pixels) ** 3 + 1e-10)
    kurtosis = np.mean((pixels - mean) ** 4) / (np.std(pixels) ** 4 + 1e-10)
    return [mean, variance, skewness, kurtosis]

#def extract_features(pixels):
    # Combine all features into one vector
#    histogram_features = extract_histogram_features(pixels, bins=10)
#    statistical_features = extract_statistical_features(pixels)
#    return np.concatenate([histogram_features, statistical_features])  # Combine 

def extract_features(pixels1, pixels2):
    # Combine all features into one vector
    mean = np.mean(pixels2) - np.mean(pixels2)
    var = np.var(pixels2) - np.var(pixels1)
    corr = np.var(pixels2) - np.var(pixels1)
    mse = np.mean((pixels2 - pixels1) ** 2)
    abs_diff = np.mean(np.abs(pixels2 - pixels1))


#    fft1, fft2 = np.fft.fft(pixels1), np.fft.fft(pixels2)
#    ffcorr = np.corrcoef(np.abs(fft2), np.abs(fft1))[0, 1]
    
    # Shape-based features
#    peak_diff = np.max(pixels2) - np.max(pixels1)

    return [mean, var, corr, mse, abs_diff]  # Combine 


def get_class(label):
    class_ignore = [0,1,4,5]
    class_agri = [2,3] #0 agri
    #class_shadow = [6] #1 shadow
    class_white = [7, 14] #2 cloud, snow
    class_nonrel = [6, 8] #3 ##non rel
    class_veg = [9]  #4 veg
    class_rel = [10, 11, 12, 13, 15] #5 rel (works..)

    l_uni = np.unique(label)
    if len(l_uni)==2:
        lu = l_uni[1]
    else:
        lu = l_uni[0]

    if lu not in class_ignore:
        if lu in class_agri:
            final_class=int(0)
#        elif lu in class_shadow:
#            final_class=int(1)
        elif lu in class_white:
            final_class=int(1)
        elif lu in class_nonrel:
            final_class=int(2)
        elif lu in class_veg:
            final_class=int(3)
        elif lu in class_rel:
            final_class=int(4)
    else:
        return 10
    return final_class

#def process_image():


def make_features_labels(dir_path):
    ids = os.listdir(dir_path + '/im1/')

    features = []
    labels = []
    for _, id in enumerate(tqdm(ids)):
        im1 = Image.open(dir_path + 'im1/{}'.format(id))
        im2 = Image.open(dir_path + 'im2/{}'.format(id))
        mask = Image.open(dir_path + 'label/{}'.format(id))

        im1 = np.array(im1) #/255.
        im2 = np.array(im2) #/255.
        mask = np.array(mask)

        cat = get_class(mask)
        if cat!=10:
            idx = np.where(mask!=0)

            #im = np.abs(im1-im2)

            for ch in range(0, 3):
                pol_pixels1 = im1[:,:,ch][idx]
                pol_pixels2 = im2[:,:,ch][idx]
                out = extract_features(pol_pixels1, pol_pixels2)
                if ch==0:
                    im_features = out
                else:
                    im_features = np.hstack((im_features, out))

            features.append(im_features)
            labels.append(cat)

    return np.array(features), np.array(labels)


X_train, y_train = make_features_labels('/home/mariapap/DATASETS/dataset_FP_v2/train/')
X_test, y_test = make_features_labels('/home/mariapap/DATASETS/dataset_FP_v2/test/')

np.save('./data_for_svm/X_train.npy', X_train)
np.save('./data_for_svm/y_train.npy', y_train)
np.save('./data_for_svm/X_test.npy', X_test)
np.save('./data_for_svm/y_test.npy', y_test)

X_train = np.load('./data_for_svm/X_train.npy')
y_train = np.load('./data_for_svm/y_train.npy')
X_test = np.load('./data_for_svm/X_test.npy')
y_test = np.load('./data_for_svm/y_test.npy')


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('shapes', X_train.shape, y_train.shape, X_test.shape, y_test.shape)




#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.15,random_state=109) # 70% training and 30% test

#Create a svm Classifier
svc = svm.SVC(kernel='rbf') # Linear Kernel


#Train the model using the training sets
svc.fit(X_train, y_train)


#Predict the response for test dataset
y_pred = svc.predict(X_test)

print('Training set score: {:.4f}'.format(svc.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(svc.score(X_test, y_test)))

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

