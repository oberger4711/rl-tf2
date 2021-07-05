import numpy as np

def make_epsilon_func_ramped(epsilon_start, epsilon_end, num_iterations, percentage_ramp):
  xp = np.array([0, percentage_ramp * num_iterations])
  fp = np.array([epsilon_start, epsilon_end])
  def func(iteration):
    if iteration < percentage_ramp * num_iterations:
      return np.interp(iteration, xp, fp)
    else:
      return epsilon_end
  return func