def dot_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
        print(f"{a[i]}*{b[i]}={result}")
    return result

def euclidean_distance(a, b):
    sum_sq = 0
    for i in range(len(a)):
        sum_sq += (a[i] - b[i]) ** 2
        print(f"{a[i]}*{b[i]}={sum_sq}")
    return sum_sq ** 0.5

def manhattan_distance(a, b):
    total = 0
    for i in range(len(a)):
        total += abs(a[i] - b[i])
    return total

def vector_norm(v):
    sum_sq = 0
    for x in v:
        sum_sq += x ** 2
        print(f"{x}**2={sum_sq}")
    return sum_sq ** 0.5

def cosine_similarity(a, b):
    dot = dot_product(a, b)
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    print(norm_a, norm_b)
    return dot / (norm_a * norm_b)

a = [1, 2, 3]
b = [2, 1, 0]

print("Dot Product:")
#print(dot_product(a, b))
print("Euclidean distance:")
#print(euclidean_distance(a, b))
print("Manhattan distance:")
#print(manhattan_distance(a, b))
print("Cosine similarity:")
print(cosine_similarity(a, b))