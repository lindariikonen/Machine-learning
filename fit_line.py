import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton


# Mouse clicker
def collect_clicks():
    plt.plot()
    coordinate_list = plt.ginput(-1, 0, True, mouse_add=MouseButton.LEFT, mouse_stop=MouseButton.RIGHT)
    x = np.array([coordinate[0] for coordinate in coordinate_list])
    y = np.array([coordinate[1] for coordinate in coordinate_list])

    return x, y


# Linear solver
def my_linfit(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    a = (sum(x*y)-(sum_x*sum_y))/(sum(x**2)-sum_x**2)
    b = sum_y-((sum(x*y)-(sum_x*sum_y))*sum_x)/(sum(x**2)-sum_x**2)
    return a, b


# Main
x, y = collect_clicks()
a, b = my_linfit(x, y)
plt.plot(x, y, 'kx')
xp = np.arange(-2, 5, 0.1)
plt.plot(xp, a*xp+b, 'r-')
print(f"My fit: a={a} and b={b}")
plt.show()
