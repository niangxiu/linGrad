[![Build Status](https://travis-ci.org/niangxiu/linGrad.svg?branch=master)](https://travis-ci.org/niangxiu/linGrad)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
<!--[![Coverage Status](https://coveralls.io/repos/github/niangxiu/linGrad/badge.svg)](https://coveralls.io/github/niangxiu/linGrad)-->


# linGrad

Initial stepsize (learning rate) in gradient descent selected by linear range.

Linear range is the range of parameter perturbations which lead to approximately linear perturbations in the states of a network.
Linear range is computed from the difference between actual perturbations in states and the tangent solution.
In linGrad, the optimal initial stepsize is such that parameter changes on all minibatches are within linear range.

For detailed explanations check the accompanying paper: https://arxiv.org/abs/1905.04561.
