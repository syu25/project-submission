import sqlite3

unique_artist_count = 'select count(distinct(artist_id)) from songs;'
unique_song_title_count = 'select count(distinct(title)) from songs;'
unique_song_count = 'select count(distinct(spotify_id)) from songs;'
unique_album_count = 'select count(distinct(album_id)) from songs;'

class SpotifyDB():

    db_name = 'spotify.db'

    def connect(self):
        return sqlite3.connect(self.db_name)

    def purge_database(self):
        con = self.connect()
        cur = con.cursor()
        cur.execute('''drop table genres''')
        cur.execute('''drop table songs''')
        cur.close()
        con.commit()
        self.create_db()


    def create_db(self):
        db = self.connect()
        cur = db.cursor()
        cur.execute('''create table if not exists songs (
                id integer not null primary key autoincrement,
                spotify_id text unique,
                artist text,
                artist_id text,
                release_date date,
                title text,
                popularity integer,
                explicit bool,
                duration integer,
                album text,
                album_type text,
                album_id text,
                danceability float,
                energy float,
                key float,
                loudness float,
                mode float,
                speechiness float,
                acousticness float,
                instrumentalness float,
                liveness float,
                valence float,
                tempo float,
                time_signature float)''')
        cur.execute('''create table if not exists genres (
            id integer not null primary key autoincrement,
            artist_id text,
            genre text,
            foreign key (artist_id) references songs(artist_id))''')
        cur.close()
        db.close()

    def insert_song(self, db, values):
        try:
            cur = db.cursor()
            cur.execute('''insert into songs(spotify_id, artist, artist_id, release_date, title, popularity, explicit, duration, album, album_type, album_id,
                danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature)
                values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', values)
        except sqlite3.IntegrityError as detail:
            print(detail)
            pass
        db.commit()
        cur.close()

    def insert_genre(self, db, id, genre):
        try:
            cur = db.cursor()
            cur.execute('''insert into genres(artist_id, genre)
                values(?,?)''', (id, genre))
        except sqlite3.IntegrityError as detail:
            print(detail)
            pass
        cur.close()

    def run_query(self, db, query):
        cur = db.cursor()
        cur.execute(query)
        row = cur.fetchall()
        cur.close()
        db.commit()
        return row

    def get_song_count(self, db):
        return self.run_query(db, unique_song_count)

    def get_title_count(self, db):
        return self.run_query(db, unique_song_title_count)

    def get_artist_count(self, db):
        return self.run_query(db, unique_artist_count)

    def get_album_count(self, db):
        return self.run_query(db, unique_album_count)