import numpy as np
import scipy as sp
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

  # would it be more efficient to have np array of arrays?
  layers_size = [np.empty(x) for x in zip(dim1, dim2)]

  # initialise bias
  for i in np.arange(nLayers + 1):
    layers_size[i][:, 0] = 1.0

  # initialise input
  layers_size[0][:, 1:] = features

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
    epsilon_init = np.sqrt(6 / np.shape(layers_size[i])[0])
    nC = np.shape(layers_size[i][1])[0]
    if i != nLayers:
      nR = np.shape(layers_size[i + 1][1])[0] - 1
    else:
      nR = np.shape(layers_size[i + 1][1])[0]
    weights_size.append(
      (np.random.uniform(0, 1, (nR * nC)) * 2.0 * epsilon_init) -
       epsilon_init
      )

  # reshape vectors into matrices
  for i in np.arange(3):
    weights_size[i] = weights_size[i].reshape([dim1[i], dim2[i]])

  ###############################################
  # initialise outcome matrix
  ###############################################
  outcomeMat = np.zeros((len(outcome), nOutcome))
  for i in np.arange(nOutcome):
    outcomeMat[:, i] = outcome == i

  ###############################################
  # return matrix templates
  ###############################################
  return([layers_size, weights_size, outcomeMat])


def forward_prop(layers, weights, nLayers): #nLayers can be inferred
  # make a template for z
  z = layers[1:len(layers)]

  # propogate through
  for i in np.arange(nLayers):
    z[i] = np.dot(layers[i], np.transpose(weights[i]))
    layers[i + 1][:, 1:layers[i + 1].shape[1]] = sigmoid(z[i])

  z[nLayers] = np.dot(layers[nLayers], np.transpose(weights[nLayers]))
  layers[nLayers + 1] = sigmoid(z[nLayers])

  return([layers, z])


def back_prop(layers_size, weights_size, penalty, outcome, unrollWeights):
  nLayers = len(layers_size) - 2
  m = outcome.shape[0]
  weights = rollParams(unrollWeights, nLayers, weights_size)

  # forward propogate
  tmp = forward_prop(layers_size, weights, nLayers)
  layers = tmp[0]
  z = tmp[1]

  ###############################################
  # cost
  ###############################################
  fn = cost(m, layers[nLayers + 1], outcome)

  # (1.0 / m) * \
  #   sum( \
  #   np.sum((-outcome) * np.log(layers[nLayers + 1]) - \
  #   (1.0 - outcome) * np.log(1.0 - layers[nLayers + 1]), \
  #   axis = 1) \
  # )

  ###############################################
  # penalty - make a seperate function for this
  ###############################################
  #penalty_term = sum([penalty / (2.0 * m) * sum(np.sum(x[:, 1:np.shape(x)[1]]**2, axis = 1)) for x in weights])

  # add penalty to cost
  fn = fn + penalty_term(m, penalty, weights)

  ###############################################
  # back propogation for errors
  ###############################################
  # copy for templates
  delta = z[:]
  gradient = weights[:]

  # errors
  delta[nLayers] = layers[nLayers + 1] - outcome
  for i in np.arange(nLayers):
    delta[i] = np.dot(delta[i + 1], weights[i + 1])[:, 1:weights[i+1].shape[1]] * sigmoidGradient(z[i])

  # gradients
  for i in np.arange(nLayers + 1):
    gradient[i] = np.dot(np.transpose(delta[i]), layers[i]) / m

    gradient[i][:, 1:gradient[i].shape[1]] = \
      gradient[i][:, 1:gradient[i].shape[1]] + (penalty / m) * \
      weights[i][:, 1:gradient[i].shape[1]]

  # unroll gradient
  gr = unrollParams(gradient)

  # return cost function and gradient
  return(fn, gr)

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
    [penalty / (2.0 * m) * sum(np.sum(x[:, 1:np.shape(x)[1]]**2, axis = 1))\
    for x in weights]\
    )
  return(tmp)

def rollParams(x, nLayers, weights_size): #nLayers can be inferred
  pos = 0
  for i in np.arange(nLayers + 1):

    dim1 = weights_size[i].shape[0]
    dim2 = weights_size[i].shape[1]
    size = weights_size[i].size

    tmp1 = pos + size
    tmp2 = np.array([x[pos:(tmp1)]])

    weights_size[i] = tmp2.reshape(dim1, dim2)
    pos += size


  return(weights_size)


def unrollParams(x):
  stretched = [mat.reshape(1, mat.size)  for mat in x]
  out = stretched[0]
  for i in np.arange(1,len(stretched)):
    out = np.append(out, stretched[i])

  return(out)

def main():

  penalty = 0.1
  data = np.loadtxt("iris.txt")
  features = data[:, 0:4]
  outcome = data[:, 4]
  outcome.astype(int)

  nLayers = 2
  templates = setup(features, outcome, nLayers, nUnits = 10, seed = 1234)
  #print(templates[2])

  # need a proper test here
  fp_test = forward_prop(templates[0], templates[1], nLayers)
  #print(fp_test[1])



  # test rolling and unrolling works
  hat = unrollParams(templates[1])
  mat = rollParams(hat, 2, templates[1])
  #print(mat ==  templates[1])

  back_prop(templates[0], templates[1], penalty, templates[2], unrollParams(templates[1]))


  # ready to test setup and functions - then to invetsigate optimisation and building the class
  # definintion


if __name__ == "__main__":
  main()
