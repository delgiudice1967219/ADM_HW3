import re
import json
from bs4 import BeautifulSoup

# Function to extract restaurant information from HTML
def extract_restaurant_info(file_path):
    # Open HTML file of current restaurant and parse it using BeautifulSoup
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

        # Create dictionary that will contains current restaurant info
        restaurant_info = {}

        # Everything we need is contained in a <div> tag that has the class below
        restaurantDetailsDiv = soup.find("div", class_="restaurant-details__components")

        # Select each nested <div> with class "row":
            # First div contains "restaurant name"
            # Second div contains address, price range, cuisine type 
            # Third div will be ignored
        mainInfo = restaurantDetailsDiv.select("div.data-sheet > div.row")

        if mainInfo[0]:
            restaurant_info['restaurantName'] = mainInfo[0].find("h1", class_="data-sheet__title").text
        if mainInfo[1]:
            indirizzo_price = mainInfo[1].select("div.data-sheet__block > div.data-sheet__block--text")

            # Split address string by comma
            indirizzoList = indirizzo_price[0].text.strip().split(",")


            # Given that different type of address can be found, we retrive data from the inverted and splitted string, so that:
                # First element is always the city
                # Second element is always the postalCode
                # Third element is always the country
                # The remaining string is the full address, so we store everything in "address" key of the dictionary
            restaurant_info['city'] = indirizzoList[-3]
            restaurant_info['postalCode'] = indirizzoList[-2]
            restaurant_info['country'] = indirizzoList[-1]
            restaurant_info['address'] = " ".join(indirizzoList[:-3]).strip().replace("\n", "")

            # Cuisine Type and Price Range are in the same string and are separated by "·", so we split them apart
            restaurant_info['priceRange'], restaurant_info['cuisineType'] = indirizzo_price[1].text.strip().split("·")

            # Price Range
            restaurant_info['priceRange'] = restaurant_info['priceRange'].strip()
            # List of cuisine types
            restaurant_info['cuisineType'] = restaurant_info['cuisineType'].strip()

        # Description
        restaurant_info['description'] = soup.find("div", class_="data-sheet__description").text.strip().replace("\n", "")

        # Facilities and Services
        facilities = soup.select("div.restaurant-details__services ul li")
        restaurant_info['facilitiesServices'] = [s.text.strip() for s in facilities]

        # Accepted Credit Cards
        credit_cards = soup.select("div.list--card img")
        restaurant_info['creditCards'] = [re.search(r"(?<=\/)[a-z]*(?=-)", c.get("data-src"))[0] for c in credit_cards]

        # Phone Number
        spansDetails = restaurantDetailsDiv.select("section.section.section-main.section__text-componets.section__text-separator div.collapse__block-title div.d-flex span")
        restaurant_info['phoneNumber'] = spansDetails[0].text.strip()

        # URL
        restaurant_info['website'] = soup.find("meta", property="og:url")["content"]

    return restaurant_info

# Function to extract restaurant information from HTML
def extract_geo_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

        # Extract information using HTML structure of the page
        restaurant_info = {}

        script_tag = soup.find("script", type="application/ld+json")

        if script_tag:
            json_content = json.loads(script_tag.string)

            restaurant_info['restaurantName'] = json_content['name']
            restaurant_info['region'] = json_content['address']['addressRegion']
            restaurant_info['latitude'] = json_content['latitude']
            restaurant_info['longitude'] = json_content['longitude']

    return restaurant_info
