from cosine_similarity import cosine_similarity
from movie_data import movie_data

def recommend(movie_title, top_n=5):
    if movie_title not in movie_data:
        print(f"'{movie_title}' not found in dataset.")
        return []

    base_vector = movie_data[movie_title]
    similarities = []

    for other_title, other_vector in movie_data.items():
        if other_title == movie_title:
            continue
        sim = cosine_similarity(base_vector, other_vector)
        similarities.append((other_title, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

movie_name = input("Enter the movie name: ")
num_movies = int(input("Enter the number of movies to recommend: "))
print(recommend(movie_name, num_movies))
