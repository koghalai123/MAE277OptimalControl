import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table


x=np.array([-2,0,2,4,6,8,10,12,14])
y=np.array([0.006,0.006,0.006,0.007,0.0075,0.0092,0.0115,0.015,0.0186])
x_train = x[:6]
y_train = y[:6]
x_test = x[6:]
y_test = y[6:]

def to_sci(x):
    return f"{x:.3e}"  


#Part 1
P = np.array([x_train**0,x_train**1]).T

P_inv = np.linalg.pinv(P)

# Least squares solution
theta = P_inv @ y_train

# Calculate least squares cost on training data
y_train_pred = P @ theta
cost_train = np.sum((y_train - y_train_pred)**2)
print("First order training cost:", cost_train)
first_order_train = cost_train

# Calculate least squares cost on test data
P_test = np.array([x_test**0, x_test**1]).T
y_test_pred = P_test @ theta
cost_test = np.sum((y_test - y_test_pred)**2)
print("First order test cost:", cost_test)
first_order_test = cost_test


#Part 2
P = np.array([x_train**0,x_train**1,x_train**2]).T
P_inv = np.linalg.pinv(P)
# Least squares solution
theta = P_inv @ y_train

# Calculate least squares cost on training data
y_train_pred = P @ theta
cost_train = np.sum((y_train - y_train_pred)**2)
print("Second order training cost:", cost_train)
second_order_train = cost_train

# Calculate least squares cost on test data
P_test = np.array([x_test**0, x_test**1, x_test**2]).T
y_test_pred = P_test @ theta
cost_test = np.sum((y_test - y_test_pred)**2)
print("Second order test cost:", cost_test)
second_order_test = cost_test


#Part 3
P = np.array([x_train**0,x_train**1,x_train**2,x_train**3]).T
P_inv = np.linalg.pinv(P)
# Least squares solution
theta = P_inv @ y_train

# Calculate least squares cost on training data
y_train_pred = P @ theta
cost_train = np.sum((y_train - y_train_pred)**2)
print("Third order training cost:", cost_train)
third_order_train = cost_train

# Calculate least squares cost on test data
P_test = np.array([x_test**0, x_test**1, x_test**2, x_test**3]).T
y_test_pred = P_test @ theta
cost_test = np.sum((y_test - y_test_pred)**2)
print("Third order test cost:", cost_test)
third_order_test = cost_test


# fourth order model
P = np.array([x_train**0,x_train**1,x_train**2,x_train**3,x_train**4]).T
P_inv = np.linalg.pinv(P)
# Least squares solution
theta = P_inv @ y_train

# Calculate least squares cost on training data
y_train_pred = P @ theta
cost_train = np.sum((y_train - y_train_pred)**2)
print("Fourth order training cost:", cost_train)
fourth_order_train = cost_train

# Calculate least squares cost on test data
P_test = np.array([x_test**0, x_test**1, x_test**2, x_test**3, x_test**4]).T
y_test_pred = P_test @ theta
cost_test = np.sum((y_test - y_test_pred)**2)
print("Fourth order test cost:", cost_test)
fourth_order_test = cost_test


# Create table of errors
def to_sci(x):
    return f"{x:.3e}"  


# Save table data and model parameters to CSV
data = {
    'Model Order': ['First', 'Second', 'Third', 'Fourth'],
    'Theta': [
        ', '.join([to_sci(v) for v in np.linalg.pinv(np.array([x_train**0, x_train**1]).T) @ y_train]),
        ', '.join([to_sci(v) for v in np.linalg.pinv(np.array([x_train**0, x_train**1, x_train**2]).T) @ y_train]),
        ', '.join([to_sci(v) for v in np.linalg.pinv(np.array([x_train**0, x_train**1, x_train**2, x_train**3]).T) @ y_train]),
        ', '.join([to_sci(v) for v in np.linalg.pinv(np.array([x_train**0, x_train**1, x_train**2, x_train**3, x_train**4]).T) @ y_train])
    ],
    'Training Error': [to_sci(first_order_train), to_sci(second_order_train), to_sci(third_order_train), to_sci(fourth_order_train)],
    'Test Error': [to_sci(first_order_test), to_sci(second_order_test), to_sci(third_order_test), to_sci(fourth_order_test)]
}
df = pd.DataFrame(data)
df.to_csv('errorTable.csv', index=False)

# Display and save table as image (without Theta column)
df_display = df.drop(columns=['Theta'])
fig, ax = plt.subplots(figsize=(6, 1.5))
ax.axis('off')
tbl = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center', cellLoc='center', colWidths=[0.35]*len(df_display.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.0, 1.2)
plt.savefig('errorTable.png', bbox_inches='tight')
plt.show()

# Plot fits for each model order with actual data
plt.figure(figsize=(4, 2.5))
plt.scatter(x, y, color='black', label='Actual Data')

# First order fit
P_full_1 = np.array([x**0, x**1]).T
theta_1 = np.linalg.pinv(np.array([x_train**0, x_train**1]).T) @ y_train
y_fit_1 = P_full_1 @ theta_1
plt.plot(x, y_fit_1, label='First Order Fit', linestyle='--')

# Second order fit
P_full_2 = np.array([x**0, x**1, x**2]).T
theta_2 = np.linalg.pinv(np.array([x_train**0, x_train**1, x_train**2]).T) @ y_train
y_fit_2 = P_full_2 @ theta_2
plt.plot(x, y_fit_2, label='Second Order Fit', linestyle='-.')

# Third order fit
P_full_3 = np.array([x**0, x**1, x**2, x**3]).T
theta_3 = np.linalg.pinv(np.array([x_train**0, x_train**1, x_train**2, x_train**3]).T) @ y_train
y_fit_3 = P_full_3 @ theta_3
plt.plot(x, y_fit_3, label='Third Order Fit', linestyle=':')

# Fourth order fit
P_full_4 = np.array([x**0, x**1, x**2, x**3, x**4]).T
theta_4 = np.linalg.pinv(np.array([x_train**0, x_train**1, x_train**2, x_train**3, x_train**4]).T) @ y_train
y_fit_4 = P_full_4 @ theta_4
plt.plot(x, y_fit_4, label='Fourth Order Fit', linestyle=(0, (3, 1, 1, 1)))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('modelPlots.png', bbox_inches='tight')
plt.show()