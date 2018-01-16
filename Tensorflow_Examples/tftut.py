import tensorflow as tf
#----------------------------
#node1 = tf.constant(3.0,tf.float32)
#node2 = tf.constant(4.0)
#sess = tf.Session()
#print(sess.run([node1,node2]))
#----------------------------
'''a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)

sess = tf.Session()
print(sess.run(f))'''
#-------------------------
T, F = 1.,-1.

bias = 1.

train_in = [[T,T,bias],[T,F,bias],[F,T,bias],[F,F,bias]]
train_out = [[T],[F],[F],[F],]

w= tf.Variable(tf.random_normal([3,1]))

def step(x):
	is_greater = tf.greater(x,0)
	as_float = tf.to_float(is_greater)
	doubled = tf.multiply(as_float,2)
	return tf.subtract(doubled,1)

output = step(tf.matmul(train_in,w))
error = tf.subtract(train_out,output)
mse = tf.reduce_mean(tf.square(error))
delta = tf.matmul(train_in,error,transpose_a=True)
train = tf.assign(w,tf.add(w,delta))
sess = tf.Session()
sess.run(tf.initialize_all_variables())
err,target = 1,0
epoch, max_epoch = 0,10
while err>target and epoch < max_epoch:
	epoch +=1
	err,_ = sess.run([mse,train])
	print(epoch,err)

feed_dict = [[T,F]]
classify = tf.run(output,feed_dict)
print(classify)




