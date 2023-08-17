import numpy as np
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length



def createFeatures(X):
  import itertools
  R = X[:,:64]
  S1 = X[:,64:68]
  S2 = X[:,68:72]

  p = list(itertools.product(*[[S1[:,:1], 1 - S1[:,:1]], [S1[:,1:2], 1 - S1[:,1:2]], [S1[:,2:3], 1 - S1[:,2:3]], [S1[:,3:4], 1 - S1[:,3:4]]]))
  q = list(itertools.product(*[[S2[:,:1], 1 - S2[:,:1]], [S2[:,1:2], 1 - S2[:,1:2]], [S2[:,2:3], 1 - S2[:,2:3]], [S2[:,3:4], 1 - S2[:,3:4]]]))

  p_list = []
  q_list = []
  for i in range(16):
    temp_p = p[i][0]*p[i][1]*p[i][2]*p[i][3]
    p_list.append(temp_p.flatten())

    temp_q = q[i][0]*q[i][1]*q[i][2]*q[i][3]
    q_list.append(temp_q.flatten())

  p_arr = np.array(p_list)
  q_arr = np.array(q_list)

  z_arr = np.subtract(p_arr, q_arr)
  z = np.transpose(z_arr)

  z_reshaped = z[:, :, np.newaxis]
  R_reshaped = R[:, np.newaxis, :]

  product = z_reshaped * R_reshaped
  num = product.shape[0]
  res = product.reshape((num,1024))

  final_dataset = np.zeros((num, 1040))
  final_dataset[:,:1024] = res
  final_dataset[:,1024:] = z

  return final_dataset

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response

    model = LinearSVC( loss = "hinge" )
    model.fit( createFeatures(Z_train[:,:72]), Z_train[:,-1] )
    
    return model				# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to make predictions on test challenges
	
	return model.predict( createFeatures(X_tst[:,:72]) )

#######
# train_data = np.loadtxt('train.dat')
# test_data = np.loadtxt('test.dat')

# model = my_fit(train_data)
# preds = my_predict(test_data, model)

# print(np.average( test_data[:,-1] == preds ))



