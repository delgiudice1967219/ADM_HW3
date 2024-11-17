import ipywidgets as ipw
import pandas as pd
from ipywidgets import HTML
from itertools import chain
from functions import engine
from IPython.display import display


class SearchRestaurantUI:
    def __init__(self, df, vocabulary_df, inverted_index):
        self.df = df
        self.vocabulary_df = vocabulary_df
        self.inverted_index = inverted_index

        # Extract unique values for each column
        self.cuisine_types = sorted(df['cuisineType'].unique().tolist())

        # We used chain to flatten the list of lists into a single list for all facilities
        # then set to remove the duplicates and eval to converts this to List
        self.all_facilities = sorted(list(set(chain(*df['facilitiesServices'].apply(eval)))))
        self.price_ranges = sorted(df['priceRange'].unique().tolist())

        # Create the widgets
        self.create_widgets()

        # Create the results output area
        self.output = ipw.Output()

    def create_widgets(self):
        # Dropdown for cuisine type selection
        self.ddCuisineType = ipw.Dropdown(
            options=['All'] + self.cuisine_types, # We added all to allow no filtering
            value='All', # Default value
            description='Cuisine:',
            disabled=False
        )

        # Checkbox for multiple facilities and services selection
        self.facilities_label = HTML(value="<b>Facilities:</b>")  # Bolded label using HTML

        # Create checkboxes for each facility
        self.facilities_checkboxes = [ipw.Checkbox(
            value=False,
            description=facility) for facility in self.all_facilities]

        # Combine the label and checkboxes in a single VBox
        self.facilities_box = ipw.VBox([self.facilities_label] + self.facilities_checkboxes)

        # Dropdown for price range selection
        self.ddPriceRange = ipw.Dropdown(
            options=['All'] + self.price_ranges, # Same as cuisineType
            value='All',
            description='Price Range:',
            disabled=False
        )

        # Text box for custom search query input
        self.txtQuery = ipw.Text(
            value='',   # Default value
            placeholder='Insert your text...',
            description='Search:',
            disabled=False
        )

        # Slider for selecting the number of results
        self.num_results = ipw.IntSlider(
            value=5,
            min=1,  # Min number of results
            max=50, # Max number of results
            step=1,
            description='Results:',
            continuous_update=False
        )

        # Search button
        self.btSearch = ipw.Button(
            description='Search',
            disabled=False,
            button_style='success',
            tooltip='Search for restaurants',
            icon='search'
        )

        # Clear button to clear all fields
        self.btClear = ipw.Button(
            description='Clear',
            disabled=False,
            button_style='danger',
            tooltip='Clear the search form'
        )

        # Callback functions to define what to do on button click
        self.btSearch.on_click(self.on_search_click)
        self.btClear.on_click(self.on_clear_click)

    def on_search_click(self, b):
        # Clear any previous results output
        self.output.clear_output()

        with self.output:
            try:
                # Collect user input into a dictionary
                query = {
                    'description': self.txtQuery.value,
                    'cuisineType': None if self.ddCuisineType.value == 'All' else self.ddCuisineType.value,
                    'facilitiesServices': [cb.description for cb in self.facilities_checkboxes if cb.value],
                    'priceRange': None if self.ddPriceRange.value == 'All' else self.ddPriceRange.value,
                    'num_results': self.num_results.value
                }

                # Call search function and get the results
                result_df = engine.find_top_custom_restaurants(
                    query,
                    self.vocabulary_df,
                    self.inverted_index,
                    self.df
                )

                if result_df.empty:
                    print("No restaurant has been found with the current selections.")
                else:
                    # Display the result pandas DataFrame
                    display(result_df)

            except Exception as e:
                print(f"Error during the search: {str(e)}")

    # Callback function for the clear button
    def on_clear_click(self, b):
        # Reset all widgets to default values
        self.txtQuery.value = ''
        self.ddCuisineType.value = 'All'
        for cb in self.facilities_checkboxes:
            cb.value = False
        self.ddPriceRange.value = 'All'
        self.num_results.value = 5

    # Display the entire UI
    def display(self):
        # Create the layout
        search_box = ipw.VBox([
            self.txtQuery,
            self.ddCuisineType,
            self.facilities_box,
            self.ddPriceRange,
            self.num_results,
            ipw.HBox([self.btSearch, self.btClear]),
            self.output
        ])
        display(search_box)