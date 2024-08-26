import unittest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn

if True:
    import sys
    sys.path.insert(
        0, 'C:/Users/Lorenzo/Desktop/Workspace/Github/Project-3/src')
    from train_model import initialize_params, forward_prop, calculate_cost, back_prop, sigmoid


class test_forward_and_back_prop(unittest.TestCase):

    def test_basic(self):
        m = 2
        X = np.array([[0.8, 0.9], [0.3, 0.5]])
        w = np.array([[0.5], [0.75]])
        b = np.array([[1.]])
        y = np.array([[1], [0]])
        params = {'w':w,'b':b}
        yhat, inter_vals = forward_prop(X, params)
        cost = calculate_cost(yhat, y)
        grads = back_prop(y, yhat, inter_vals, X)
        dw_result = grads['dw']
        db_result = grads['db']

        X = torch.tensor(X)
        y = torch.tensor(y)
        w = torch.tensor(w, requires_grad=True)
        b = torch.tensor(b, requires_grad=True)
        z = (X @ w + b).clone().detach().requires_grad_(True)
        yhat = (1/(1 + torch.exp(-z))).clone().detach().requires_grad_(True)
        losses = (y * torch.log(yhat)) + ((1 - y) * torch.log(1 - yhat))
        cost = -torch.sum(losses, dim=0, keepdims=True)/m
        print(f'cost torch {cost}')
        cost.backward()
        print(f'dyhat {yhat.grad.detach()}')
        print(f'dz_dw {z.grad.deta}')
        dw_expected = w.grad.detach().numpy()
        db_expected = b.grad.detach().numpy()
        dyhat = yhat.grad
        
        
        np.testing.assert_allclose(dw_result, dw_expected)
        np.testing.assert_allclose(db_result, db_expected)


unittest.main()
