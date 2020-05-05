import sqlite3
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
import seaborn as sns
import random


def create_graph(x, y, typefun, title, xlab, ylab, color ='#1DB954', fns = None, size = None, save = True):
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
    plt.show()

db = sqlite3.connect('spotify.db')

songs = pd.read_sql_query('select * from songs where energy != 0 and popularity != 0;', db)

ordered_genres = pd.read_sql_query('select genre, artist_id from (select distinct artist_id, genre from genres) order by count(*) over (partition by genre) desc', db)
random.seed(64)
shuffled_genres = ordered_genres.sample(frac=1)
shuffled_genres['first_id'] = shuffled_genres['artist_id'].mask(shuffled_genres['artist_id'].duplicated())
one_genre = shuffled_genres.dropna()

songs_with_genres = songs.set_index('artist_id').join(one_genre.set_index('artist_id')).set_index('spotify_id')

features = songs_with_genres.drop(['id', 'popularity', 'artist', 'release_date', 'title', 'popularity',
     'album', 'album_id', 'album_type', 'explicit', 'duration', 'genre', 'first_id', 'mode', 'time_signature', 'key'], axis = 1)

scale = preprocessing.MinMaxScaler().fit_transform

features[['loudness']] = scale(songs[['loudness']].values)
features[['tempo']] = scale(songs[['tempo']].values)

# print(features)

# rss_varying_k = [ KMeans(n_clusters = k).fit(features).inertia_ for k in range(1, 10)]

# print(rss_varying_k)
# create_graph(range(1,10), rss_varying_k, plt.plot, 'Determining Best K', 'k', 'RSS')

fit = KMeans(n_clusters = 4).fit(features)
pc = PCA(n_components = 2).fit_transform(features)
pd_pc = pd.DataFrame(pc)
pd_pc['class'] = fit.predict(features)
pd_pc.columns = ['x', 'y', 'class']

songs_with_genres['kmeans_class'] = pd_pc['class'].values

# snplot = sns.lmplot(data = pd_pc, hue = 'class', x = 'x', y = 'y',
#                    fit_reg = False, legend = True, legend_out = True)
# snplot.fig.set_facecolor('black')
# plt.ylabel('Y', color='white')
# plt.xlabel('X', color='white')
# plt.title('Title', color='white')
# ax = plt.gca()
# ax.set_facecolor('black')
# ax.spines['bottom'].set_color('white')
# ax.spines['top'].set_color('white')
# ax.spines['left'].set_color('white')
# ax.spines['right'].set_color('white')
# ax.tick_params(axis='y', colors='white')
# ax.tick_params(axis='x', colors='white')
# plt.show()

indices = songs_with_genres[songs_with_genres['genre']  == 'pop'].index
indices2 = songs_with_genres[songs_with_genres['genre']  == 'pop rap'].index
removed_pop = songs_with_genres.drop(indices).drop(indices2)

# i = 0
# for group in removed_pop.groupby('kmeans_class'):
#     value_counts = group[1]['genre'].value_counts()
#     wordcloud = WordCloud(max_words = 1000, margin = 10, width = 1600, height = 900)
#     wordcloud.generate_from_frequencies(frequencies = value_counts)
#     fig = plt.figure()
#     plt.imshow(wordcloud.recolor(colormap='Greens'), interpolation='bilinear')
#     plt.axis('off')
#     plt.show()
#     fig.savefig(f'wordcloud{i}.png', facecolor = fig.get_facecolor())
#     i += 1

features['class'] = pd_pc['class'].values


print(features[features['class'] == 0].mean())
print(features[features['class'] == 1].mean())
print(features[features['class'] == 2].mean())
print(features[features['class'] == 3].mean())
