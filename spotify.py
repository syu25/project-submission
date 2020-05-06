import spotipy
from threading import Thread
from spotipy.oauth2 import SpotifyClientCredentials
from kivy.app import App
from database import SpotifyDB

MAX_SEARCH_RESULTS = 2000
CLIENT_ID = 'ENTER CLIENT ID'
CLIENT_SECRET = 'ENTER CLIENT SECRET'

default_audio_features = {
    'danceability': None,
    'energy': None,
    'key': None,
    'loudness': None,
    'mode': None,
    'speechiness': None,
    'acousticness': None,
    'instrumentalness': None,
    'liveness': None,
    'valence': None,
    'tempo': None,
    'time_signature': None,
}

class SpotifyScraper():
    
    running = False
    saved_query = ''
    last_offset = 0
    scrape_thread = False
    db = SpotifyDB()
    db.create_db()

    def toggle_scraper(self, stats, query, year):

        try_query = f'{query} year:{year}' if year >= 0 and year <= 2020 else query
        if len(try_query) == 0:
            return False

        self.running = not self.running

        if self.running:
            self.saved_query = try_query
            self.scrape_thread = Thread(target = self.scrape_songs, args = (stats,))
            self.scrape_thread.start()

        return True

    def purge(self):
        self.db.purge_database()

    def reset_search(self):
        self.saved_query = ''
        self.last_offset = 0

    def song_count(self):
        con = self.db.connect()
        count = self.db.get_song_count(con)[0][0]
        con.close()
        return count

    def title_count(self):
        con = self.db.connect()
        count = self.db.get_title_count(con)[0][0]
        con.close()
        return count

    def artist_count(self):
        con = self.db.connect()
        count = self.db.get_artist_count(con)[0][0]
        con.close()
        return count

    def album_count(self):
        con = self.db.connect()
        count = self.db.get_album_count(con)[0][0]
        con.close()
        return count

    def scrape_songs(self, stats):
        client_credentials_manager = SpotifyClientCredentials(CLIENT_ID, CLIENT_SECRET)
        spotify = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
        stopped = False
        print(self.saved_query)
        con = self.db.connect()

        for i in range(self.last_offset, MAX_SEARCH_RESULTS, 50):
            self.last_offset = i
            if not self.running:
                stopped = True
                break

            results = spotify.search(q = self.saved_query, limit = 50, offset = i)
            song_ids = [ song['id'] for song in results['tracks']['items'] ]
            
            if len(song_ids) == 0:
                break

            audio_features = spotify.audio_features(song_ids)
            album_details = [ (song['album']['name'], song['album']['album_type'], song['album']['id']) for song in results['tracks']['items'] ]
            artist_ids = [ song['artists'][0]['id'] for song in results['tracks']['items'] ]
            artists = spotify.artists(artist_ids)

            artist_details = [ ( artist['id'], artist['genres'] ) for artist in artists['artists'] ]

            for _, (t, f, al, ar) in enumerate(zip(results['tracks']['items'], audio_features, album_details, artist_details)):
                f = f or default_audio_features
                details = (t['id'], t['artists'][0]['name'], ar[0], t['album']['release_date'], t['name'], t['popularity'], t['explicit'], t['duration_ms'], 
                    al[0], al[1], al[2],
                    f['danceability'], f['energy'], f['key'], f['loudness'], f['mode'], f['speechiness'], f['acousticness'], f['instrumentalness'],
                    f['liveness'], f['valence'], f['tempo'], f['time_signature'])

                self.db.insert_song(con, details)

                for genre in (ar[1] or []):
                    self.db.insert_genre(con, ar[0], genre)

            if results['tracks']['next'] is None:
                break

            stats['song_count'] = self.song_count()
            stats['song_title_count'] = self.title_count()
            stats['album_count'] = self.album_count()
            stats['artist_count'] = self.artist_count()

        con.close()

        if not stopped:
            self.running = False
            self.saved_query = ''
            self.last_offset = 0
            stats['scraping'] = False
            stats['has_query'] = False
            App.get_running_app().reset_query_animation()
