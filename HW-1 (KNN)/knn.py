from collections import Counter
import numpy as np
def euclidean_dis(x1,x2): 
  return np.sqrt(np.sum((x1 - x2)** 2)) #Euclidean
def manhattan_dis(x1,x2): #Manhattan
  return np.sum(np.abs(x1-x2))

class k_nn:
  def __init__(knn, k=7, dis_calc="euclidean"):  #Initilaziton
    knn.k = k
    knn.dis_calc = dis_calc
  def train (knn,x_train,y_train): #Train
    knn.x_train = x_train
    knn.y_train = y_train
  def result (knn,x_test): #Final Result
    results = [knn._predict(x) for x in x_test]
    return np.array(results)
  def _predict (knn,x): # Prediction Mechanism
    if knn.dis_calc == "euclidean":
      distances = [euclidean_dis(x,x_train) for x_train in knn.x_train]
    else:
      distances = [manhattan_dis(x,x_train) for x_train in knn.x_train]

    k_index = np.argsort(distances)[:knn.k]
    k_nearest = [knn.y_train[i] for i in k_index]
    most_common = Counter(k_nearest).most_common(1)
    return most_common[0][0]




    
