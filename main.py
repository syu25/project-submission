from kivy import Config
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, StringProperty, BooleanProperty, ListProperty, DictProperty
from kivy.clock import Clock
from threading import Thread
from kivy.animation import Animation
from os.path import dirname, join
import time
from spotify import SpotifyScraper
from graphs import SpotifyGraphs
import os

class SpotifyScreen(Screen):

    def add_widget(self, *args):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args)
        return super(SpotifyScreen, self).add_widget(*args)

class SpotifyApp(App):

    Config.set('graphics', 'minimum_width', '800')
    Config.set('graphics', 'minimum_height', '600')

    graphs = SpotifyGraphs()
    scraper = SpotifyScraper()

    # Navigation UI

    index = NumericProperty(-1)
    screens = []
    screen_names = ListProperty([])

    # Populate Songs UI

    stats = DictProperty({
        'song_count': scraper.song_count(),
        'song_title_count': scraper.title_count(),
        'album_count': scraper.album_count(),
        'artist_count': scraper.artist_count(),
        'scraping': False,
        'has_query': False
    })
    current_title = StringProperty()
    status_text = StringProperty('Populate The Database')

    # Genre Wordcloud UI
    wordcloud_disabled = BooleanProperty(False)
    wordcloud_source = StringProperty('data/icons/placeholder.png')

    # Determine Best K UI
    kmeans_disabled = BooleanProperty(False)
    kmeans_source = StringProperty('data/icons/placeholder.png')

    # Determine Best K UI
    pca_disabled = BooleanProperty(False)
    pca_source = StringProperty('data/icons/placeholder.png')


    def build(self):
        curdir = dirname(__file__)
        self.title = 'Spotify Song Exploration'
        self.screens = {}
        self.screen_names = ['Populate Songs', 'Genre Wordcloud', 'KMeans', 'PCA', 'Purge Data']
        self.available_screens = [ join(curdir, 'screens', f'{fn}.kv') for fn in self.screen_names ]
        self.next_screen()

    def on_pause(self):
        return True

    def on_resume(self):
        pass

    def prev_screen(self):
        self.index = (self.index - 1) % len(self.available_screens)
        screen = self.load_screen(self.index)
        sm = self.root.ids.sm
        sm.switch_to(screen, direction = 'right')
        self.current_title = screen.name

    def next_screen(self):
        self.index = (self.index + 1) % len(self.available_screens)
        screen = self.load_screen(self.index)
        sm = self.root.ids.sm
        sm.switch_to(screen, direction = 'left')
        self.current_title = screen.name

    def load_screen(self, index):
        if index in self.screens:
            return self.screens[index]
        screen = Builder.load_file(self.available_screens[index])
        self.screens[index] = screen
        return screen

    def toggle_scraper(self, query, year = 0):

        try:
            year = int(year)
        except ValueError:
            year = -1

        if not self.scraper.toggle_scraper(self.stats, query, year):
            close_popup = Button(text = 'Dismiss')
            popup_label = Label(text = 'Enter a search term or a valid year!')
            popup_layout = BoxLayout(orientation = 'vertical')
            popup_layout.add_widget(popup_label)
            popup_layout.add_widget(close_popup)
            popup = Popup(title='Query Error', content = popup_layout,
                size_hint = (None, None), size = (400, 200))
            close_popup.bind(on_press = popup.dismiss)
            popup.open()
            return

        self.stats.scraping = self.scraper.running
        self.stats.has_query = True
        self.reset_query_animation()

    def reset_search(self):
        self.scraper.reset_search()
        self.stats.has_query = False
        self.reset_query_animation()

    def reset_query_animation(self):
        height = 48 if self.stats.has_query else 0
        Animation(height = height, duration = .3).start(self.screens[0].ids.query_info)

    def try_purge(self):
        confirm = Button(text = 'Yes', background_color = [0,1,0,1])
        cancel = Button(text = 'No', background_color = [1,0,0,1])
        popup_layout = BoxLayout(orientation = 'horizontal')
        popup_layout.add_widget(confirm)
        popup_layout.add_widget(cancel)
        popup = Popup(title='Are you sure?', content = popup_layout,
            size_hint = (None, None), size = (400, 100))
        cancel.bind(on_press = popup.dismiss)
        confirm.bind(on_press = self.confirm_purge)
        confirm.bind(on_press = popup.dismiss)
        popup.open()

    def confirm_purge(self, args):
        self.scraper.purge()
        self.stats['song_count'] = 0
        self.stats['song_title_count'] = 0
        self.stats['album_count'] = 0
        self.stats['artist_count'] = 0

    def generate_wordcloud(self):
        def create_image():
            self.wordcloud_disabled = True
            def set_loading(args):
                self.wordcloud_source = 'data/icons/loading.zip'

            Clock.schedule_once(set_loading)
            path = self.graphs.genre_wordcloud()
            def set_image(args):
                self.wordcloud_source = path
                self.wordcloud_disabled = False
            Clock.schedule_once(set_image)

        image_thread = Thread(target = create_image)
        image_thread.start()

    def determine_best_k(self):
        def create_image():
            self.kmeans_disabled = True
            def set_loading(args):
                self.kmeans_source = 'data/icons/loading.zip'

            Clock.schedule_once(set_loading)
            path = self.graphs.kmeans_selection()
            def set_image(args):
                self.kmeans_source = path
                self.kmeans_disabled = False
            Clock.schedule_once(set_image)

        image_thread = Thread(target = create_image)
        image_thread.start()

    def show_pca(self, count):
        try:
            count = int(count)
        except ValueError:
            count = 2

        def create_image():
            self.pca_disabled = True
            def set_loading(args):
                self.pca_source = 'data/icons/loading.zip'

            Clock.schedule_once(set_loading)
            path = self.graphs.pca(count)
            def set_image(args):
                self.pca_source = path
                self.pca_disabled = False
            Clock.schedule_once(set_image)

        image_thread = Thread(target = create_image)
        image_thread.start()

if __name__ == '__main__':
    SpotifyApp().run()