import numpy as np

vocabulary_file = 'word_embeddings.txt'


def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def find_analogy(x, y, z, words, vectors):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    target_vec = z + (y - x)

    distances = []
    for i, vec in enumerate(vectors):
        dist = euclidean_distance(target_vec, vectors[vec])
        distances.append((words[i], dist))

    # Sort by distance in ascending order (lower distance means more similar)
    distances.sort(key=lambda x: x[1])

    return distances[:2]


# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding='utf8') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding='utf8') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

# Main loop for analogy
while True:
    x, y, z = input("\nEnter three words separated with space: ").split()

    # use z = z + (y − x) and find word closest to the new z
    vector_x = vectors[x]
    vector_y = vectors[y]
    vector_z = vectors[z]

    targets = find_analogy(vector_x, vector_y, vector_z, words, vectors)

    for target, distance in targets:
        print(f"\n\"{x} is to {y} like {target} is to {z}\"")
    break
