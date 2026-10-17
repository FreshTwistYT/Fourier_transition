import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc


class Integrator:
    
    def __init__(self):

        self.weights_NC = [
            [1, 1, 2],                
            [1, 4, 1, 6],        
            [1, 3, 3, 1, 8],            
            [7, 32, 12, 32, 7, 90],      
            [19, 75, 50, 50, 75, 19, 288] 
        ]
        # В __init__ добавляем относительные узлы для каждого порядка от 0 до 1:
        self.nodes_NC = [
            np.array([0.0, 1.0]),               
            np.array([0.0, 0.5, 1.0]),           
            np.array([0.0, 1/3, 2/3, 1.0]),      
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ]


        self.weights_Gauss = [
            [2.0],
            [1.0, 1.0],
            [0.555555555555556, 0.888888888888888, 0.555555555555556],
            [0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454],
            [0.236926885056189, 0.478628670499367, 0.568888888888889, 0.478628670499367, 0.236926885056189]
        ]

        self.nodes_Gauss = [
            [0.0],
            [-0.577350269189626, 0.577350269189626],
            [-0.774596669241483, 0.0, 0.774596669241483],
            [-0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053],
            [-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664]
        ]

    def Newton_Cotes(self, func, a, b, order, count):
        cin = np.array(self.weights_NC[order-1][:order+1])
        Cn = self.weights_NC[order-1][-1]
        
        # Берем заранее созданный шаблон узлов
        nodes = self.nodes_NC[order - 1] 
        
        H = (b - a) / count
        I = 0
        for i in range(count):
            block_beginning = a + i * H
            x_nodes = block_beginning + H * nodes 
            y_values = func(x_nodes)
            I += H * np.sum(cin * y_values) / Cn
        return I


    def Gauss(self, func, a, b, order, count):
        # Метод Гаусса 
        # a, b - границы интегрирования
        # count - число подынтервалов
        # func - интегрируемая функция

        nodes = np.array(self.nodes_Gauss[order - 1])
        weights = np.array(self.weights_Gauss[order - 1])
        H = (b - a) / count

        I = 0

        for i in range(count):
            block_beginning = a + i * H
            block_end = block_beginning + H

            x_nodes = (H / 2) * nodes + (block_beginning + block_end) / 2

            y_values = func(x_nodes)
            I += (H / 2) * np.sum(weights * y_values)   

        return I