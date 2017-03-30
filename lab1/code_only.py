import time
import math
import numpy as np
import matplotlib.pyplot as plt



------------------------------------------------------
Разделение точек на плоскости, аналитическое решение


def generate_cloud(center, count):
    points = []
    for i in range(count):
        point = np.random.normal(0.0, 3.0, 2)
        points.append(point + center)
    return points


def solve(cloud1, cloud2):
    matrix = cloud1 + cloud2
    n = len(cloud1)
    # we add 1 to each row of the matrix for computing b=w[0] in line equation
    matrix = [[1] + row.tolist() for row in matrix]
    pseudo_inverse_matrix = np.linalg.pinv(matrix)
    labels = [1 if i < n else -1 for i in range(2 * n)]
    return np.dot(pseudo_inverse_matrix, labels)


def draw_plot(cloud1, cloud2, w):
    plt.scatter([p[0] for p in cloud1], [p[1] for p in cloud1], c='g')
    plt.scatter([p[0] for p in cloud2], [p[1] for p in cloud2], c='r')
    plt.plot([x for x in range(-7, 15)], [(-w[0] - x * w[1]) / w[2] for x in range(-7, 15)], 'k-')
    plt.show()


def main():
    n = 250
    cloud1 = generate_cloud(np.array([0, 0]), n)
    cloud2 = generate_cloud(np.array([6, 6]), n)
    w = solve(cloud1, cloud2)
    draw_plot(cloud1, cloud2, w)


if __name__ == '__main__':
    main()
    
    
    
    

------------------------------------------------------
Градиентный спуск на функции x^2+xy+y^2




def f(point):
    x, y = point[0], point[1]
    return x ** 2 + x * y + y ** 2


def gradient(point):
    x, y = point[0], point[1]
    return np.array([2 * x + y, 2 * y + x])


def render(levels, path):
    # creating array of function values
    x = np.arange(-3, 3, 0.025)
    y = np.arange(-3, 3, 0.025)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    levels.sort()
    plt.contour(X, Y, Z, levels)
    # drawing algorithm's path
    plt.plot([p[0] for p in path], [p[1] for p in path], 'k-')


def gradient_descend(start, path, levels):
    cur_point, prev_point = start, start + 1
    step_rate, eps = 0.1, 1 / 1e6
    while math.fabs(f(cur_point) - f(prev_point)) > eps:
        prev_point = cur_point.copy()
        cur_point -= gradient(cur_point) * step_rate
        path.append(cur_point.copy())
        levels.append(f(cur_point))
    return cur_point


def main():
    start_point = np.array([1.9, 1.5])
    path, levels = [], []
    gradient_descend(start_point, path, levels)
    render(levels, path)
    plt.show()

if __name__ == '__main__':
    main()
    
    
    
    
    
    
------------------------------------------------------
Градиентный спуск на функции Розенброка с эвристикой






EPS = 1 / 1e10

def f(point):
    x, y = point[0], point[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def gradient(point):
    x, y = point[0], point[1]
    return np.array([-2 * (1 - x) - 400 * (y - x ** 2) * x, 200 * (y - x ** 2)])


def render(f, bounds, levels, path):
    # creating array of function values
    delta = 0.025
    x = np.arange(bounds['min_x'], bounds['max_x'], delta)
    y = np.arange(bounds['min_y'], bounds['max_y'], delta)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    levels.sort()
    plt.contour(X, Y, Z, levels, colors='#89D2A9')
    # drawing algorithm's path
    plt.plot([p[0] for p in path], [p[1] for p in path], 'k-')


def make_step(f, cur_point, direction):
    step_length = 1e5
    while f(cur_point) < f(cur_point - direction * step_length):
        step_length /= 2.99
    cur_point -= direction * step_length
    return cur_point


def gradient_descend(f, start, path, levels):
    path.append(start.copy())
    levels.append(f(start))
    prev_value = 1e18
    cur_point = start

    while math.fabs(f(cur_point) - prev_value) > EPS:
        prev_value = f(cur_point)
        point1 = cur_point.copy()
        cur_point = make_step(f, cur_point, gradient(cur_point))
        cur_point = make_step(f, cur_point, gradient(cur_point))
        cur_point = make_step(f, cur_point, (point1 - cur_point))
        path.append(cur_point.copy())
        levels.append(f(cur_point))

    return cur_point


start_point = np.array([8.9, 13.5])
bounds = {'min_x': -5, 'max_x': 2, 'min_y': -1, 'max_y': 16}
path, levels = [], []
print(gradient_descend(f, start_point, path, levels))
render(f, bounds, levels, path)
plt.show()






------------------------------------------------------
Поиск разделяющей прямой с помощью градиентного спуска







def generate_cloud(center, count):
    points = []
    for i in range(count):
        point = np.random.normal(0.0, 3.0, 2)
        points.append(point + center)
    return points


def gradient_descend(f, gradient, start_point):
    cur_point, prev_point = start_point, start_point + 1
    step_rate, eps = 1 / 1e5, 1 / 1e6
    while math.fabs(f(cur_point) - f(prev_point)) > eps:
        prev_point = cur_point.copy()
        cur_point -= gradient(cur_point) * step_rate
    return cur_point


def get_strict_solution(matrix, labels):
    pseudo_inverse_matrix = np.linalg.pinv(matrix)
    return np.dot(pseudo_inverse_matrix, labels)


def loss_f_gradient(w, x, y):
    grad = [0, 0, 0]
    for i in range(len(x)):
        dot_product = np.dot(x[i], w)
        grad[0] += 2 * (dot_product - y[i])
        grad[1] += 2 * (dot_product - y[i]) * x[i][1]
        grad[2] += 2 * (dot_product - y[i]) * x[i][2]
    return np.array(grad)


def loss_f(w, x, y):
    ans = 0
    for i in range(len(x)):
        ans += (np.dot(w, x[i]) - y[i]) ** 2
    return ans


def solve(matrix, labels):
    start_point = np.random.uniform(2, 10, 3)
    solution = gradient_descend(
        lambda point: loss_f(point, matrix, labels),
        lambda point: loss_f_gradient(point, matrix, labels),
        start_point)
    return solution / np.linalg.norm(solution)


def draw_plot(cloud1, cloud2, w1, w2):
    plt.scatter([p[0] for p in cloud1], [p[1] for p in cloud1], c='g')
    plt.scatter([p[0] for p in cloud2], [p[1] for p in cloud2], c='r')
    plt.plot([x for x in range(-10, 20)], [(-w1[0] - x * w1[1]) / w1[2] for x in range(-10, 20)], 'b')
    plt.plot([x for x in range(-10, 20)], [(-w2[0] - x * w2[1]) / w2[2] for x in range(-10, 20)], 'r')
    plt.show()


def main():
    n = 250
    cloud1 = generate_cloud(np.array([0, 0]), n)
    cloud2 = generate_cloud(np.array([6, 6]), n)
    matrix = cloud1 + cloud2
    # we add 1 to each row of the matrix for computing b=w[0] in line equation
    matrix = [[1] + row.tolist() for row in matrix]
    labels = [1 if i < n else -1 for i in range(2 * n)]
    time_start = time.clock()
    w = solve(matrix, labels)
    print("Gradient descends time: ", time.clock() - time_start)
    w2 = get_strict_solution(matrix, labels)
    w /= np.linalg.norm(w)
    w2 /= np.linalg.norm(w2)
    print('Accurate solution: ', w2)
    print('Gradient descends solution: ', w)
    print('Difference: ', w2 - w)
    draw_plot(cloud1, cloud2, w, w2)


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
----------------------------------------------------
Сравнение обычного градиентного спуска и наискорейшего






def generate_cloud(center, count):
    points = []
    for i in range(count):
        point = np.random.normal(0.0, 3.0, 2)
        points.append(point + center)
    return points


# if param quickest is false, it would be classical gradient descend
def gradient_descend(f, gradient, start_point, quickest, path):
    cur_point, prev_point = start_point, start_point + 1
    step_rate, eps = 1 / 1e5, 1 / 1e6
    while math.fabs(f(cur_point) - f(prev_point)) > eps:
        prev_point = cur_point.copy()
        grad = gradient(cur_point)
        if quickest:
            step_rate = 4
            while f(cur_point) < f(cur_point - grad * step_rate):
                step_rate /= np.random.uniform(3, 4, 1)
            step_rate /= 2
        cur_point -= grad * step_rate
        path.append(f(prev_point))
    return cur_point


def get_strict_solution(matrix, labels):
    pseudo_inverse_matrix = np.linalg.pinv(matrix)
    return np.dot(pseudo_inverse_matrix, labels)


def loss_f_gradient(w, x, y):
    grad = [0, 0, 0]
    for i in range(len(x)):
        dot_product = np.dot(x[i], w)
        grad[0] += 2 * (dot_product - y[i])
        grad[1] += 2 * (dot_product - y[i]) * x[i][1]
        grad[2] += 2 * (dot_product - y[i]) * x[i][2]
    return np.array(grad)


def loss_f(w, x, y):
    ans = 0
    for i in range(len(x)):
        ans += (np.dot(w, x[i]) - y[i]) ** 2
    return ans


def solve(start_point, matrix, labels, quickest, gradient_descend_path):
    solution = gradient_descend(
        lambda point: loss_f(point, matrix, labels),
        lambda point: loss_f_gradient(point, matrix, labels),
        start_point, quickest, gradient_descend_path)
    return solution / np.linalg.norm(solution)


def draw_plot(cloud1, cloud2, w1, w2):
    plt.scatter([p[0] for p in cloud1], [p[1] for p in cloud1], c='g')
    plt.scatter([p[0] for p in cloud2], [p[1] for p in cloud2], c='r')
    plt.plot([x for x in range(-10, 20)], [(-w1[0] - x * w1[1]) / w1[2] for x in range(-10, 20)], 'b')
    plt.plot([x for x in range(-10, 20)], [(-w2[0] - x * w2[1]) / w2[2] for x in range(-10, 20)], 'r')
    plt.show()


def main():
    n = 250
    cloud1 = generate_cloud(np.array([0, 0]), n)
    cloud2 = generate_cloud(np.array([6, 6]), n)
    matrix = cloud1 + cloud2
    # we add 1 to each row of the matrix for computing b=w[0] in line equation
    matrix = [[1] + row.tolist() for row in matrix]
    labels = [1 if i < n else -1 for i in range(2 * n)]
    start_point = np.random.uniform(2, 10, 3)
    path, path2 = [], []
    # quickest descend
    solve(start_point.copy(), matrix, labels, 1, path)
    plt.plot([i for i in range(len(path))], [y for y in path], 'r')
    # classical descend
    solve(start_point.copy(), matrix, labels, 0, path2)
    plt.plot([i for i in range(len(path2))], [y for y in path2], 'b')
    axes = plt.gca()
    axes.set_xlim([0, 600])
    axes.set_ylim([0, 5e3])
    plt.xlabel('Step')
    plt.ylabel('Loss function value')
    plt.show()


if __name__ == '__main__':
    main()
    
    
    
    
    
    
----------------------------------------------------
Стохастический градиентный спуск на mnist





def parse_data():
    f = open('train.csv', 'r')
    s = ' '
    m, k = 0, 0
    labels, matrix = [], []
    while s:
        s = f.readline()
        if len(s) > 0 and (s[0] == '0' or s[0] == '1'):
            matrix.append(list(map(int, s[2:].split(','))))
            labels.append(int(s[0]))
        k += 1
    f.close()
    f = open('train.txt', 'w')
    for i in range(int(len(matrix) * 0.8)):
        f.write(str(labels[i]) + ',' + ','.join(map(str, matrix[i])) + '\n')
    f.close()
    f = open('check.txt', 'w')
    for i in range(int(len(matrix) * 0.8), len(matrix)):
        f.write(str(labels[i]) + ',' + ','.join(map(str, matrix[i])) + '\n')
    f.close()


def read_data(file_name):
    labels, matrix = [], []
    f = open(file_name, 'r')
    s = f.readline()
    while s:
        matrix.append(list(map(int, s[2:].split(','))))
        labels.append(int(s[0]))
        s = f.readline()
    f.close()
    matrix = [[1] + row for row in matrix]
    labels = list(map(lambda x: x if x == 1 else -1, labels))
    return {'matrix': matrix, 'labels': labels}


def loss_f(w, x, y):
    ans = 0.
    for i in range(len(x)):
        ans += math.log(1 + math.exp(-y[i] * np.dot(x[i], w)))
    return ans


def loss_f_gradient(w, x, y):
    grad = [0] * len(w)
    for i in range(len(x)):
        e = math.exp(-y[i] * np.dot(x[i], w))
        for j in range(len(w)):
            grad[j] += -y[i] * x[i][j] * e / (1 + e)
    return np.array(grad)


def gradient_descend(f, gradient, start_point, matrix, labels):
    eps, step_rate = 1 / 1e8, 1 / 1e6
    cur_point, prev_point = start_point.copy(), start_point.copy()
    prev_point += 0.1
    cur, size = 0, 3
    batch, batch_labels = matrix[cur:cur + size], labels[cur:cur + size]

    while math.fabs(np.linalg.norm(cur_point - prev_point)) > eps:
        prev_point = cur_point.copy()
        cur_point -= gradient(cur_point, batch, batch_labels) * step_rate
        cur += size
        if cur > len(labels):
            cur = 0
        batch, batch_labels = matrix[cur:cur + size], labels[cur:cur + size]

    return cur_point


def solve(matrix, labels):
    start_point = np.array([0.] * len(matrix[0]))
    w = gradient_descend(loss_f, loss_f_gradient, start_point, matrix, labels)
    return w


def measure_quality(w, matrix, labels):
    cnt = 0
    for i in range(len(matrix)):
        if labels[i] * np.dot(w, matrix[i]) >= 0:
            cnt += 1
    return cnt / len(matrix)


def main():
    data = read_data('train.txt')
    test_data = read_data('check.txt')
    w = solve(data['matrix'], data['labels'])
    print('Accuracy: ', measure_quality(w, test_data['matrix'], test_data['labels']))


# parse_data()

if __name__ == "__main__":
    main()
    
    
    
    
    

----------------------------------------------
Momentum





def f(point):
    x, y = point[0], point[1]
    return 10 * x ** 2 + y ** 2


def gradient(point):
    x, y = point[0], point[1]
    return np.array([20 * x, 2 * y])


def render(levels, path, path_color):
    # creating array of function values
    x = np.arange(-1.5, 1.5, 0.0025)
    y = np.arange(-1, 2, 0.0025)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    levels.sort()
    plt.contour(X, Y, Z, levels)
    # drawing algorithm's path
    plt.plot([p[0] for p in path], [p[1] for p in path], path_color)


def gradient_descend(start, path, levels):
    cur_point, prev_point = start, start + 1
    step_rate, eps = 0.1, 1 / 1e6
    while math.fabs(f(cur_point) - f(prev_point)) > eps:
        path.append(cur_point.copy())
        levels.append(f(cur_point))
        prev_point = cur_point.copy()
        cur_point -= gradient(cur_point) * step_rate
    return cur_point


def gradient_descend_with_momentum(start, path, levels):
    cur_point, prev_point = start, start + 1
    step_rate, eps, l = 0.1, 1 / 1e6, 0.68
    dx = np.array([0, 0])
    while math.fabs(f(cur_point) - f(prev_point)) > eps:
        path.append(cur_point.copy())
        levels.append(f(cur_point))
        prev_point = cur_point.copy()
        dx = (1 - l) * dx + l * gradient(cur_point) * step_rate
        cur_point -= dx
    return cur_point


def main():
    start_point = np.array([1.24, 1.35])
    path, levels, path2, levels2 = [], [], [], []
    gradient_descend(start_point.copy(), path, levels)
    gradient_descend_with_momentum(start_point.copy(), path2, levels2)
    render(levels, path, 'b')
    render(levels2, path2, 'r')
    plt.show()

if __name__ == '__main__':
    main()
