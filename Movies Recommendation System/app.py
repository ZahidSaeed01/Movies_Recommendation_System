from flask import Flask,render_template,url_for,request
import pandas as pd
from imdb import IMDb
import Knn

ia = IMDb()

movies_dataframe = pd.read_csv('movies.csv')
links_dataframe = pd.read_csv('links.csv')
ratings_dataframe = pd.read_csv('ratings.csv')
tags_dataframe = pd.read_csv('tags.csv')
processedDataframe = pd.read_csv('processed_data.csv')

recommender = Knn.KNN(movies_dataframe,links_dataframe,ratings_dataframe,tags_dataframe)

recommender.processedData = processedDataframe


def get_movies(imdb_ids):
  movies_data = []
  for id in imdb_ids:
    movie = ia.get_movie(id)
    movies_data.append(movie)
  
  return movies_data



app = Flask(__name__)


@app.route("/")
def index():
  movie_id = int(request.args.get('movie'))

  imdb_ids = recommender.recommend_movies(movie_id)
  movie = ia.get_movie(movie_id)
  movies = get_movies(imdb_ids)

  genres = ''
  for genre in movie['genres']:
    genres += " / " + genre
  
  movie['genres'] = genres

  return render_template('index.html',current_movie=movie,recommended_movies=movies)


if __name__ == "__main__":
  app.run(debug=True)
