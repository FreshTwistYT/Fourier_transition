import numpy as np


class Integrator:
    
    def __init__(self):

        self.weights_NC = [
            [1, 1, 2],
            [1, 4, 1, 6],
            [1, 3, 3, 1, 8],
            [7, 32, 12, 32, 7, 90],
            [19, 75, 50, 50, 75, 19, 288]
        ]

    def Newton_Cotes(self, func, a, b, order, count):
        # Метод Ньютона-Котеса 
        # a, b - границы интегрирования
        # count - число подынтервалов
        # func - интегрируемая функция
        cin = np.array(self.weights_NC[order-2][:order])
        Cn = self.weights_NC[order-2][-1]
        H = (b - a) / count
        h = H / order

        I = 0

        for i in range(count):
            block_beginning = a + i * H
            x_nodes = np.array([block_beginning + p * h for p in range(order)])
            y_values = func(x_nodes)
            I_local = H * np.sum(cin * y_values) / Cn
            I += I_local

        return I