import ipywidgets as ipw
import pandas as pd
from itertools import chain
import functions
from IPython.display import display

class SearchRestaurantUI:
    def __init__(self, df, vocabulary_df, inverted_index):
        self.df = df
        self.vocabulary_df = vocabulary_df
        self.inverted_index = inverted_index

        # Estrai i valori unici per ciascuna colonna
        self.cuisine_types = sorted(df['cuisineType'].unique().tolist())
        self.all_facilities = sorted(list(set(chain(*df['facilitiesServices'].apply(eval))))) # Chain fa diventare tutte le liste di stringhe ovvero la colonna una lista unica
        self.price_ranges = sorted(df['priceRange'].unique().tolist())

        # Crea i widget
        self.create_widgets()

        # Output per i risultati
        self.output = ipw.Output()

    def create_widgets(self):
        # Dropdown per il tipo di cucina
        self.ddCuisineType = ipw.Dropdown(
            options=['All'] + self.cuisine_types,
            value='All',
            description='Cuisine Type:',
            disabled=False
        )

        # Checkbox per selezione multipla dei servizi
        self.facilities_checkboxes = [ipw.Checkbox(value=False, description=facility) for facility in self.all_facilities]
        self.facilities_box = ipw.VBox(self.facilities_checkboxes)

        # Dropdown per la fascia di prezzo
        self.ddPriceRange = ipw.Dropdown(
            options=['All'] + self.price_ranges,
            value='All',
            description='Price Range:',
            disabled=False
        )

        # Text box per inserire il testo di ricerca
        self.txtQuery = ipw.Text(
            value='',
            placeholder='Insert your text...',
            description='Text to search:',
            disabled=False
        )

        self.num_results = ipw.IntSlider(
            value=5,
            min=1,
            max=50,
            step=1,
            description='Results:',
            continuous_update=False
        )

        # Bottone di ricerca
        self.btSearch = ipw.Button(
            description='Search',
            disabled=False,
            button_style='success',
            tooltip='Search for restaurants',
            icon='search'
        )

        # Bottone per pulire i campi
        self.btClear = ipw.Button(
            description='Clear',
            disabled=False,
            button_style='danger',
            tooltip='Clear the search form'
        )

        # Collega i bottoni alle funzioni di callback
        self.btSearch.on_click(self.on_search_click)
        self.btClear.on_click(self.on_clear_click)

    def on_search_click(self, b):
        # Cancella l'output precedente
        self.output.clear_output()

        with self.output:
            try:
                # Raccolta dei valori inseriti dall'utente
                query = {
                    'description': self.txtQuery.value,
                    'cuisineType': None if self.ddCuisineType.value == 'All' else self.ddCuisineType.value,
                    'facilitiesServices': [cb.description for cb in self.facilities_checkboxes if cb.value],
                    'priceRange': None if self.ddPriceRange.value == 'All' else self.ddPriceRange.value,
                    'num_results': self.num_results.value
                }

                # Chiamata alla funzione di ricerca
                result_df = functions.find_top_custom_restaurants(
                    query,
                    self.vocabulary_df,
                    self.inverted_index,
                    self.df
                )

                if result_df.empty:
                    print("No restaurant has been found with the current selections.")
                else:
                    # Visualizzazione del numero di risultati selezionato dall'utente
                    display(result_df.head(query['num_results']))

            except Exception as e:
                print(f"Error during the search: {str(e)}")

    def on_clear_click(self, b):
        self.txtQuery.value = ''
        self.ddCuisineType.value = 'All'
        for cb in self.facilities_checkboxes:
            cb.value = False
        self.ddPriceRange.value = 'All'
        self.num_results.value = 5

    def display(self):
        # Creazione del layout dell'interfaccia
        search_box = ipw.VBox([
            self.txtQuery,
            self.ddCuisineType,
            ipw.Label('Facilities:'),
            self.facilities_box,
            self.ddPriceRange,
            self.num_results,
            ipw.HBox([self.btSearch, self.btClear]),
            self.output
        ])
        display(search_box)