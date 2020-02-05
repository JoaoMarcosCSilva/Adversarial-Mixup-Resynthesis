def interpolate(code_a, code_b, t):
  return code_a*(1-t) + (t)*code_b

def linear(codes, t = None):
  length = tf.shape(codes)[0]
  if t is None:
    t = tf.random.uniform((length,1,1,1), 0, 0.5)
  return interpolate(codes, codes[::-1], t), t
  