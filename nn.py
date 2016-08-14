import numpy as np
from scipy.optimize import minimize
import copy


class NNet:
  '''
  Represents a neural net
  '''

  def __init__(self, input, outcome, nLayers = 2, nUnits = 10, penalty = 0.1):
    '''
    should define n hidden layers
    n units in output layer
    input is numpy array
    outcome is numpy array (a vector)
    '''
    return()


  def __str__(self):
    '''
    decide what a suitible string representation is
    '''
    return()



  def predict(self, input):
    '''
    predict method
    '''
    return()


  def get_weights(self):
    '''
    accessor for weights
    '''
    return()



def setup(features, outcome, nLayers, nUnits, seed = 1234):
  np.random.seed(seed)
  nFeature = features.shape[1]
  nOutcome =  len(set(outcome))
  if nOutcome == 2: nOutcome = 1
  nSample = features.shape[0]

  ###############################################
  # initialise dimensions of layers
  ###############################################
  dim1 = [nSample] * (nLayers + 2)
  dim2 = [nFeature + 1]
  dim2 += ([nUnits + 1] * nLayers)
  dim2.append(nOutcome)

  a_size = [np.empty(x) for x in zip(dim1, dim2)]

  # initialise bias
  for i in np.arange(nLayers + 1):
    a_size[i][:, 0] = 1.0

  # initialise input
  a_size[0][:, 1:] = features

  ###############################################
  # initialise dimensions of weights (weights)
  ###############################################
  dim1 = dim2[:]
  dim1 = dim1[1:]
  tmp1 = len(dim1)
  dim1 = [dim1[x] - 1 if x < (tmp1 - 1) else dim1[x] for x in np.arange(tmp1)]
  #
  dim2 = dim2[:-1]

  # initialise vector of correct length to be reshaped
  weights_size = list()
  for i in np.arange(nLayers + 1):
    epsilon_init = np.sqrt(6 / np.shape(a_size[i])[0])
    nC = np.shape(a_size[i][1])[0]
    if i != nLayers:
      nR = np.shape(a_size[i + 1][1])[0] - 1
    else:
      nR = np.shape(a_size[i + 1][1])[0]
    weights_size.append(
      (np.random.uniform(0, 1, (nR * nC)) * 2.0 * epsilon_init) -
       epsilon_init
      )

  # reshape vectors into matrices
  for i in np.arange(3):
    weights_size[i] = weights_size[i].reshape([dim1[i], dim2[i]], order='F')

  ###############################################
  # initialise outcome matrix
  ###############################################
  outcomeMat = np.zeros((len(outcome), nOutcome))
  for i in np.arange(nOutcome):
    outcomeMat[:, i] = outcome == i

  ###############################################
  # return matrix templates
  ###############################################
  return([a_size, weights_size, outcomeMat])


def forward_prop(activations, weights):
  # hidden layers- subtract input and output
  nLayers = len(activations) - 2
  # make a template for z
  #z = copy.deepcopy(layers[1:len(layers)])
  z = activations[1:len(activations)][:]

  # propogate through
  for i in np.arange(nLayers):
    z[i] = np.dot(activations[i], np.transpose(weights[i]))
    activations[i + 1][:, 1:activations[i + 1].shape[1]] = sigmoid(z[i])

  z[nLayers] = np.dot(activations[nLayers], np.transpose(weights[nLayers]))
  activations[nLayers + 1] = sigmoid(z[nLayers])

  return([activations, z])


def back_prop(unrollWeights, a_size, weights_size, penalty, outcome):
  nLayers = len(a_size) - 2
  m = outcome.shape[0]
  weights = rollParams(unrollWeights, weights_size)

  ###############################################
  # forward propogation for cost and penalty
  ###############################################
  tmp = forward_prop(a_size, weights)
  layers = tmp[0]
  z = tmp[1]

  # cost and penalty term
  fn = cost(m, layers[nLayers + 1], outcome)
  fn += penalty_term(m, penalty, weights)

  # CORRECT UP TO HERE: gradients differ however.

  ###############################################
  # back propogation for errors
  ###############################################

  # copy for templates
  delta = z[:]
  gradient = weights[:]

  # errors
  delta[nLayers] = layers[nLayers + 1] - outcome
  for i in np.arange(nLayers-1, -1, -1):
    delta[i] = np.dot(delta[i + 1], weights[i + 1])[:, 1:weights[i+1].shape[1]] \
    * sigmoidGradient(z[i])

  # gradients
  for i in np.arange(nLayers + 1):
    gradient[i] = np.dot(np.transpose(delta[i]), layers[i]) / m

    gradient[i][:, 1:gradient[i].shape[1]] += (penalty / m) * \
      weights[i][:, 1:gradient[i].shape[1]]


  # unroll gradient
  gr = unrollParams(gradient)

  # return cost function and gradient
  return(fn, gr)



def bp(a_size, weights_size, penalty, outcome):
  nLayers = len(a_size) - 2
  m = outcome.shape[0]
  gradient = weights_size[:]
  delta = a_size[1:len(a_size)]

  def bp_inner(unrollWeights):
    weights = rollParams(unrollWeights, weights_size)
    ###############################################
    # forward propogation for cost and penalty
    ###############################################
    tmp = forward_prop(a_size, weights)
    activations = tmp[0]
    z = tmp[1]

    # cost and penalty term
    fn = cost(m, activations[nLayers + 1], outcome)
    fn += penalty_term(m, penalty, weights)

    ###############################################
    # back propogation for errors
    ###############################################

    # copy for templates
    delta = z[:]
    #gradient = weights[:]

    # errors
    delta[nLayers] = activations[nLayers + 1] - outcome
    for i in np.arange(nLayers-1, -1, -1):
      delta[i] = np.dot(delta[i + 1], weights[i + 1])[:, 1:weights[i+1].shape[1]] \
      * sigmoidGradient(z[i])

    # gradients
    for i in np.arange(nLayers + 1):
      gradient[i] = np.dot(np.transpose(delta[i]), activations[i]) / m

      gradient[i][:, 1:gradient[i].shape[1]] += (penalty / m) * \
        weights[i][:, 1:gradient[i].shape[1]]

    # unroll gradient
    gr = unrollParams(gradient)

    # return cost function and gradient
    return(fn, gr)

  return(bp_inner)



def sigmoid(x):
  return(1.0 / (1.0 + np.exp(-x)))

def sigmoidGradient(x):
  return(sigmoid(x) * (1.0 - sigmoid(x)))

def cost(m, outcome_layer, outcome):
  tmp = (1.0 / m) * \
    sum(\
    np.sum((-outcome) * np.log(outcome_layer) - \
    (1.0 - outcome) * np.log(1.0 - outcome_layer), axis = 1) \
    )
  return(tmp)



def penalty_term(m, penalty, weights):
  tmp = sum(\
    [(penalty / (2.0 * m)) * sum(np.sum(x[:, 1:np.shape(x)[1]]**2, axis = 1))\
    for x in weights]\
    )
  return(tmp)

def rollParams(x, weights_size):
  nLayers = len(weights_size) - 1
  pos = 0
  for i in np.arange(nLayers + 1):
    dim1 = weights_size[i].shape[0]
    dim2 = weights_size[i].shape[1]
    size = weights_size[i].size
    tmp1 = pos + size
    tmp2 = np.array([x[pos:(tmp1)]])
    weights_size[i] = tmp2.reshape(dim1, dim2, order='F')
    pos += size

  return(weights_size)


def unrollParams(x):
  stretched = [mat.reshape(1, mat.size, order='F')  for mat in x]
  out = stretched[0]
  for i in np.arange(1,len(stretched)):
    out = np.append(out, stretched[i])

  return(out)


# how to do a closure:
def power(x):
  def power_inner(y):
    return(y ** x)
  return(power_inner)

def main():

  penalty = 0.01
  data = np.loadtxt("iris.txt")
  data2 = np.loadtxt("nneptrWeights.dat")
  features = data[:, 0:4]
  outcome = data[:, 4]
  outcome.astype(int)

  nLayers = 2

  # checked vs nneptr: all templates correct shape
  templates = setup(features, outcome, nLayers, nUnits = 20, seed = 123)
  a_size = templates[0][:]
  weights_size = templates[1][:]
  outcome_mat = templates[2][:]

  # roll / unroll tested and correct:

  #######################################
  # test
  weights_size = rollParams(data2, weights_size)
  init_weights = unrollParams(weights_size)

  #print(weights_size)
  #print(init_weights)

  # need a proper test here
  # seems reasonable though
 #fp_test = forward_prop(a_size, weights_size)



  #cost, grad = back_prop(init_weights, a_size, weights_size, penalty, outcome_mat)
  #print(grad)
  #print(cost)

  bp_closure = bp(a_size, weights_size, penalty, outcome_mat)

  #cost, grad = back_prop(templates[0], templates[1], penalty, templates[2], unrollParams(templates[1]))


 # print(cost, grad)


  opt_weights = minimize(
  fun = bp_closure,
  x0 = init_weights,
  method = 'L-BFGS-B',
  jac = True
  )


  # now for scipy optimisation:
  # opt_weights = minimize(
  #   fun = back_prop,
  #   x0 = init_weights,
  #   method = 'L-BFGS-B',
  #   jac = True,
  #   args = (a_size,\
  #   weights_size,\
  #   penalty,\
  #   outcome_mat)
  #   )

  #print(opt_weights.x)

  final_weights = rollParams(opt_weights.x, weights_size)
  #print(final_weights)


  fp_final = forward_prop(a_size, final_weights)

  print(fp_final[0])

  bloop = power(3)
  print(bloop(2))


if __name__ == "__main__":
  main()
