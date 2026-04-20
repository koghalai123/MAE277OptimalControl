import numpy as np



x=np.array([-2,0,2,4,6,8,10,12,14])
y=np.array([0.006,0.006,0.006,0.007,0.0075,0.0092,0.0115,0.015,0.0186])
x_train = x[:6]
y_train = y[:6]
x_test = x[6:]
y_test = y[6:]


#Part 1
P = np.array([x_train**0,x_train**1]).T

P_inv = np.linalg.pinv(P)

# Least squares solution
theta = P_inv @ y_train

# Calculate least squares cost on training data
y_train_pred = P @ theta
cost_train = np.sum((y_train - y_train_pred)**2)
print("First order training cost:", cost_train)

# Calculate least squares cost on test data
P_test = np.array([x_test**0, x_test**1]).T
y_test_pred = P_test @ theta
cost_test = np.sum((y_test - y_test_pred)**2)
print("First order test cost:", cost_test)


#Part 2
P = np.array([x_train**0,x_train**1,x_train**2]).T

P_inv = np.linalg.pinv(P)

# Least squares solution
theta = P_inv @ y_train

# Calculate least squares cost on training data
y_train_pred = P @ theta
cost_train = np.sum((y_train - y_train_pred)**2)
print("Second order training cost:", cost_train)

# Calculate least squares cost on test data
P_test = np.array([x_test**0, x_test**1, x_test**2]).T
y_test_pred = P_test @ theta
cost_test = np.sum((y_test - y_test_pred)**2)
print("Second order test cost:", cost_test)


#Part 3
P = np.array([x_train**0,x_train**1,x_train**2,x_train**3]).T

P_inv = np.linalg.pinv(P)

# Least squares solution
theta = P_inv @ y_train

# Calculate least squares cost on training data
y_train_pred = P @ theta
cost_train = np.sum((y_train - y_train_pred)**2)
print("Third order training cost:", cost_train)

# Calculate least squares cost on test data
P_test = np.array([x_test**0, x_test**1, x_test**2, x_test**3]).T
y_test_pred = P_test @ theta
cost_test = np.sum((y_test - y_test_pred)**2)
print("Third order test cost:", cost_test)

print("done")