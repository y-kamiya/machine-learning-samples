from gensim import corpora, models, similarities

def create_similarity_matrix(movie_features):
    dictionary = corpora.Dictionary(movie_features)
    bow_actors = [dictionary.doc2bow(actors) for actors in MOVIES]
    tfidf = models.TfidfModel(bow_actors)
    similarity_matrix = similarities.MatrixSimilarity(tfidf[bow_actors])
    return dictionary, similarity_matrix

def calc_similarity(dictionary, similarity_matrix, user_features):
    return [
            (movie_index, similarity)
            for movie_index, similarity
            in enumerate(similarity_matrix[dictionary.doc2bow(user_features)])
            ]

MOVIES = [
    ['actor1', 'actor2', 'actor3'],
    ['actor3', 'actor5', 'actor7'],
    ['actor4', 'actor5', 'actor9'],
    ['actor1', 'actor3', 'actor6'],
    ['actor2', 'actor5', 'actor7'],
    ['actor1', 'actor4', 'actor6'],
    ['actor3', 'actor5', 'actor8'],
    ['actor5', 'actor6', 'actor7'],
    ['actor1', 'actor7', 'actor9'],
    ['actor2', 'actor6', 'actor8'],
]

USER = ['actor3', 'actor5', 'actor7'],

dictionary, similarity_matrix = create_similarity_matrix(MOVIES)
similarity = calc_similarity(dictionary, similarity_matrix, USER)
print(similarity)
