from integrating import Integrator
import numpy as np
import matplotlib.pyplot as plt
import time

f = lambda x: np.sin(x)
a = 0
b = np.pi
int = 2

Integrator = Integrator()

xs = [i for i in range(100, 1000, 10)]
errNC = [[], [], [], [], []]
errG = [[], [], [], [], []]
timeNC = [[], [], [], [], []]
timeG = [[], [], [], [], []]
memNC = [[], [], [], [], []]
memG = [[], [], [], [], []]

for o in range(1,6):
    for x in xs:
        st = time.perf_counter()
        INC = Integrator.Newton_Cotes(f, a, b, o, x)
        fn = time.perf_counter()
        errNC[o-1].append(abs(int-INC))
        timeNC[o-1].append(fn-st)

        st = time.perf_counter()
        IG = Integrator.Gauss(f, a, b, o, x)
        fn = time.perf_counter()
        errG[o-1].append(abs(int-IG))
        timeG[o-1].append(fn-st)

    plt.plot(xs, errNC[o-1], label='Ньютон-Котес', color='red')
    plt.plot(xs, errG[o-1], label='Гаусс', color='black')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('Число разбиений')
plt.ylabel('Абсолютная ошибка')
plt.yscale('log')
plt.show()

for o in range(5):
    plt.plot(xs, timeNC[o], label='Ньютон-Котес', color='red')
    plt.plot(xs, timeG[o], label='Гаусс', color='black')   
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.xlabel('Число разбиений')
plt.ylabel('Время выполнения')
plt.legend(by_label.values(), by_label.keys())
plt.show()
