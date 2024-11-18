def euclidean_distance(a,b):
    return float(sum((x-y) ** 2 for x,y in zip(a,b)) ** 0.5)

def k_nn(x, minority, k):
    distances = []
    for x_i in minority:
        if x_i != x:
            distance = euclidean_distance(x,x_i)
            distances.append((distance, x_i))

    distances.sort(key=lambda x:x[0])
    nn = [x_i for _, x_i in distances[:k]]
    return nn
