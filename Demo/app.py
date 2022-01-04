import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
import scipy.stats
from fcmeans import FCM
from PIL import Image
from sklearn.metrics import silhouette_score

def VisualImages(lst_image, lst_title=None):
    plt.figure(figsize=(15,15))
    plt.axis("off")
    n = len(lst_image)
    if lst_title==None:
        for i in range(n):
            plt.subplot(1, n, i+1)
            lst_image[i] = cv.cvtColor(lst_image[i], cv.COLOR_BGR2RGB)
            lst_image[i] = plt.imshow(lst_image[i], interpolation="bicubic")
    else:
        for i in range(n):
            plt.subplot(1, n, i+1)
            lst_image[i] = cv.cvtColor(lst_image[i], cv.COLOR_BGR2RGB)
            [i] = plt.imshow(lst_image[i], interpolation="bicubic")
            plt.title(lst_title[i])
    plt.show()

import time

def algo_kmeans(X, k, cal_score):
  score = None
  k_means = KMeans(n_clusters = k, random_state = 0).fit(X)
  tt = k_means.cluster_centers_[k_means.labels_]
  timee = 0
  if cal_score:
    t1 = time.time()
    score = silhouette_score(X, k_means.labels_)
    t2 = time.time()
    timee = t2 - t1
  return tt, score, timee
def algo_fcm(X, k, cal_score):
  score = None
  fcm = FCM(n_clusters=k)
  fcm.fit(np.array(X))
  fcm_labels = fcm.predict(np.array(X))
  tt = fcm.centers[fcm_labels]
  timee = 0
  if cal_score:
    t1 = time.time()
    score = silhouette_score(X, fcm_labels)
    t2 = time.time()
    timee = t2 - t1
  return tt, score, timee
def algo_meanshift(X,k,cal_score):
  score = None
  bandwidth_=estimate_bandwidth(X, quantile=0.2, n_samples=500)
  if k!=0:
    bandwidth_=k
 
  ms = MeanShift(bandwidth=bandwidth_, bin_seeding=True)
  ms.fit(X)
  tt = ms.cluster_centers_[ms.labels_]
  timee = 0
  if cal_score:
    t1 = time.time()
    score = silhouette_score(X, ms.labels_)
    t2 = time.time()
    timee = t2 - t1
  return tt, score, timee
def algo_em(X, k,cal_score):
  score = None
  em = GaussianMixture(n_components=k, covariance_type='full', max_iter=20, random_state=0).fit(X)
  centers = np.empty(shape=(em.n_components, X.shape[1]))
  for i in range(em.n_components):
    density = scipy.stats.multivariate_normal.logpdf(X,cov=em.covariances_[i], mean=em.means_[i], allow_singular=True)
    centers[i, :] = X[np.argmax(density)]
  pred = em.predict(X)
  tt = centers[pred]
  timee = 0
  if cal_score:
    t1 = time.time()
    score = silhouette_score(X, pred)
    t2 = time.time()
    timee = t2 - t1
  return tt, score, timee

def feature_input(coor, img):
  nrow, ncol,nchl = img.shape
  g = []
  if coor ==0:
     g = img.reshape(nrow*ncol,nchl)
  elif coor==1:
    for i in range(nrow):
      for j in range(ncol):
        temp = [img[i][j][0], img[i][j][1], img[i][j][2], i, j]
        g.append(temp)
  elif coor==2: 
    for i in range(nrow):
      for j in range(ncol):
        temp = [img[i][j][0], img[i][j][1], img[i][j][2], int(i/nrow*255), int(i/ncol*255)]
        g.append(temp)
  g = np.array(g)
  return g 

def ClusteringAlgorithms(img, option='km', k=6, coor=0, cal_score=False):
  if img.shape[1] <=250:
    img_t = img
  else:
    dim = (int(img.shape[1] * 250/ img.shape[0]), 250)
    img_t = cv.resize(img, dim, cv.INTER_AREA)
  nrow, ncol,nchl = img_t.shape
  g = feature_input(coor, img_t)

  if option=='km':
    tt, score_res, timee = algo_kmeans(g, k, cal_score)

  elif option=='fcm': #fuzzy cmeans
    tt, score_res, timee = algo_fcm(g, k, cal_score)

  elif option=='ms': #meanshift
    tt, score_res, timee = algo_meanshift(g,k, cal_score)

  elif option=='em': #Expectation Maximization
    tt, score_res, timee = algo_em(g, k, cal_score)
  
  t = tt[:,0:3]
  img_res = t.reshape(nrow, ncol, nchl)
  img_res = cv.resize(img_res, (img.shape[1], img.shape[0]), cv.INTER_NEAREST)
  return img_t, img_res, score_res, timee

def Visual_Result(img, algo, coor, cal_score, k=6):
  img_t, img_res, score_res, timee = ClusteringAlgorithms(img, algo, k, coor, cal_score)
  print("Time to evaluate:", timee, "s")
  print("Silhouette Coefficient:", score_res)
  print()
  VisualImages([img, img_res.astype('uint8')], ['Origin', 'After clustering'])

st.header("Image Segmentation with Clustering")
st.sidebar.title("Upload image")
uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

chon_thuat_toan = st.sidebar.radio("Chọn loại thuật toán",("Kmeans", "Fuzzy C-Means", "Mean shift", "Em"), key='chon_thuat_toan')

k = st.sidebar.slider("Chọn k: ", min_value=0, max_value=100, step=1)

chon_loai_dac_trung = st.sidebar.radio("Chọn loại đặc trưng đầu vào: ", ("RBG", "RBGXY", "RBGX'Y'"), key="chon_loai_dac_trung")

danh_gia = st.sidebar.radio("Tính Silhouette Score? ", ("Có", "Không"), key="danh_gia")

if uploaded_file==None:
    st.sidebar.write("Thêm ảnh!")
else:
    img = cv.imread("./image/" + uploaded_file.name)
    st.header("Origin")
    st.image(img, channels="BGR")


 
if st.sidebar.button("Click here") and uploaded_file:
        algo = ""
        if chon_thuat_toan=="Kmeans":
            algo = "km"
        elif chon_thuat_toan=="Fuzzy C-Means":
            algo = "fcm"
        elif chon_thuat_toan=="Mean shift":
            algo = "ms"
        elif chon_thuat_toan=="Em":
            algo = "em"
        yesno_danhgia = False
        if danh_gia == "Có":
          yesno_danhgia = True
        else:
          yesno_danhgia = False

        st.header(chon_loai_dac_trung)
        if chon_loai_dac_trung=="RBG":
            coor=0
            img_t, img_res, score_res, timee = ClusteringAlgorithms(img, option=algo, k=k, coor=0, cal_score=yesno_danhgia)
            st.image(img_res.astype("uint8"), channels="BGR", clamp=True )
        elif chon_loai_dac_trung=="RBGXY":
            coor=1
            img_t, img_res, score_res, timee = ClusteringAlgorithms(img, option=algo, k=k, coor=1, cal_score=yesno_danhgia)
            st.image(img_res.astype("uint8"), channels="BGR", clamp=True )
        elif chon_loai_dac_trung=="RBGX'Y'":
            coor=2
            img_t, img_res, score_res, timee = ClusteringAlgorithms(img, option=algo, k=k, coor=2, cal_score=yesno_danhgia)
            st.image(img_res.astype("uint8"), channels="BGR", clamp=True )
        st.caption("Silhouette Score: "+ str(score_res))
        st.caption("Time to evaluate: "+str(timee)+" s")



        _,img_af1,_,_ = ClusteringAlgorithms(img, option=algo, k=k, coor=0)
        _,img_af2,_,_ = ClusteringAlgorithms(img, option=algo, k=k, coor=1)
        _,img_af3,_,_ = ClusteringAlgorithms(img, option=algo, k=k, coor=2)
         
        # st.image(img_af.astype("uint8"), channels="BGR", clamp=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("RBG")
            st.image(img_af1.astype("uint8"), channels="BGR", clamp=True )
        with col2:
            st.header("RBGXY")
            st.image(img_af2.astype("uint8"), channels="BGR", clamp=True )
        with col3:
            st.header("RBGX'Y'")
            st.image(img_af3.astype("uint8"), channels="BGR", clamp=True )



