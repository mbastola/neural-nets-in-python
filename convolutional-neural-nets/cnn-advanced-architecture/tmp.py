import tensorflow as tf

logits = [ [ 10, 500, 5000,5001 -1, 0.5, 12 ] ]
top = tf.argmax(logits, 1)

top2 = tf.argmax(tf.nn.softmax(logits), 1)

sess = tf.Session()
a = sess.run(top)
b = sess.run(top2)
print(top, a)
print(top2, b)
