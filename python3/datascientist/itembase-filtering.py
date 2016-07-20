import numpy
from scipy.spatial.distance import cosine

def calc_item_score(target_index, rating_matrix):
    target_ratings = rating_matrix[target_index]
    item_similarity = numpy.zeros(len(target_ratings))
    for index in range(len(rating_matrix)):
        ratings = rating_matrix[index]
        if index == target_index:
            continue
        user_similarity = 1.0 - cosine(target_ratings, ratings)
        item_similarity += user_similarity * ratings
    return item_similarity

example_rating = numpy.array([
    [5, 3, 0, 0],
    [4, 0, 4, 1],
    [1, 1, 0, 5],
    [0, 0, 4, 4],
    [0, 1, 5, 4],
])

predict_ratings = calc_item_score(0, example_rating)
print(predict_ratings)
