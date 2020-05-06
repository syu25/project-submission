import sqlite3
import random
import pandas as pd
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')
placeholder_path = 'data/icons/placeholder.png'

def create_graph(x, y, typefun, title, xlab, ylab, color ='#1DB954', fns = None, size = None):
    fig = plt.figure(figsize = size)
    fig.set_facecolor('black')
    typefun(x, y, figure = fig, color = color)
    plt.ylabel(ylab, color='white')
    plt.xlabel(xlab, color='white')
    plt.title(title, color='white')
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')
    if(fns):
        fns()
    path = f'data/temp/{title}.png'
    fig.savefig(path, facecolor = fig.get_facecolor())
    return path

class SpotifyGraphs():

    def genre_wordcloud(self):

        con = sqlite3.connect('spotify.db')
        path = 'data/temp/genre_wordcloud.png'
        genres = pd.read_sql_query('select count(genre) as counts, genre as words from (select distinct artist_id, genre from genres) group by genre order by count(genre) desc', con)
        con.close()
        word_dict = { k : v for v, k in genres.values}
        
        if len(word_dict) == 0:
            return placeholder_path

        wordcloud = WordCloud(max_words = 1000, margin = 10, width = 1600, height = 900)
        wordcloud.generate_from_frequencies(frequencies = word_dict)
        wordcloud.recolor(colormap='Greens')
        wordcloud.to_file(path)
        return path

    def get_song_features(self):
        db = sqlite3.connect('spotify.db')
        songs = pd.read_sql_query('select * from songs where energy != 0 and popularity != 0;', db)

        if len(songs) == 0:
            return []

        ordered_genres = pd.read_sql_query('select genre, artist_id from (select distinct artist_id, genre from genres) order by count(*) over (partition by genre) desc', db)
        random.seed(64)
        shuffled_genres = ordered_genres.sample(frac=1)
        shuffled_genres['first_id'] = shuffled_genres['artist_id'].mask(shuffled_genres['artist_id'].duplicated())
        one_genre = shuffled_genres.dropna()
        songs_with_genres = songs.set_index('artist_id').join(one_genre.set_index('artist_id')).set_index('spotify_id')
        scale = preprocessing.MinMaxScaler().fit_transform
        features = songs_with_genres.drop(['id', 'popularity', 'artist', 'release_date', 'title', 'popularity',
            'album', 'album_id', 'album_type', 'explicit', 'duration', 'genre', 'first_id', 'mode', 'time_signature', 'key'], axis = 1)

        features[['loudness']] = scale(songs[['loudness']].values)
        features[['tempo']] = scale(songs[['tempo']].values)
        return features

    def kmeans_selection(self):
        features = self.get_song_features()
        if len(features) == 0:
            return placeholder_path

        rss_varying_k = [ KMeans(n_clusters = k).fit(features).inertia_ for k in range(1, 10)]
        return create_graph(range(1,10), rss_varying_k, plt.plot, 'Determining Best K', 'k', 'RSS')

    def pca(self, k):
        path = f'data/temp/PCA-{k}.png'
        features = self.get_song_features()
        
        if len(features) == 0:
            return placeholder_path

        pc = PCA(n_components = 2).fit_transform(features)
        fit = KMeans(n_clusters = k).fit(features)
        pd_pc = pd.DataFrame(pc)
        pd_pc['class'] = fit.predict(features)
        pd_pc.columns = ['x', 'y', 'class']
        snplot = sns.lmplot(data = pd_pc, hue = 'class', x = 'x', y = 'y',
                   fit_reg = False, legend = True, legend_out = True)
        snplot.fig.set_facecolor('black')
        plt.ylabel('PC2', color='white')
        plt.xlabel('PC1', color='white')
        plt.title('Visualization of Clusters', color='white')
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')
        snplot.savefig(path, facecolor = snplot.fig.get_facecolor())
        plt.close()
        return path
        