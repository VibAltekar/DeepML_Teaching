import numpy as np
import matplotlib.pyplot as plt
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
def forward(x):
   return x * w

def loss(x, y):
   y_pred = forward(x)
   return (y_pred - y) * (y_pred - y)

def gradient(x, y):  # d_loss/d_w
   return 2 * x * (x * w - y)

print("predict (before training)",  4, forward(4))

# Training loop
w_list = []
mse_list = []
for epoch in range(100):
   l_sum = 0
   for x_val, y_val in zip(x_data, y_data):
       grad = gradient(x_val, y_val)
       w = w - 0.01 * grad
       print("\tgrad: ", x_val, y_val, grad)
       l = loss(x_val, y_val)
       l_sum += l
   w_list.append(w)
   mse_list.append(l_sum / 3)

   print("progress:", epoch, "w=", w, "loss=", l)

# After training
print("predict (after training)",  "4 hours", forward(4))


plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
