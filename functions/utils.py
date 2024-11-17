import heapq
import os
import re
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from unidecode import unidecode
import string
import unicodedata
import pandas as pd
#from nltk.corpus import wordnet as wn
from collections import defaultdict
#nltk.download('wordnet')
#nltk.download('omw-1.4')
import re
import pickle
import numpy as np
import os
import glob
import pandas as pd
import re
from bs4 import BeautifulSoup
from functions import engine
from functions import parser

# Function that iterate over folders to scrape each file HTML
def iterate_folders(output_dir):
    # List to store dictionary data current restaurant
    restaurants_data = []

    # Iterate over each restaurant folder and scrape each HTML page
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        
        # Check that path is a directory
        if os.path.isdir(folder_path):
            # Iterate over files in the current folder
            for filename in os.listdir(folder_path):
                # Check if current file is .html, if so scrape the page
                if filename.endswith(".html"):  
                    print(f"Processing: {filename}")
                    file_path = os.path.join(folder_path, filename)
                    restaurant_info = extract_restaurant_info(file_path)
                    # Add dictionary data to the list of all restaurant data
                    restaurants_data.append(restaurant_info)

    return restaurants_data

def iterate_geo_folders(output_dir):

    # List to store restaurant data
    restaurants_data = []

    # Loop through all files in the directory and extract information
    for page_num in range(1, 101):
        # Build directory path for each page to scrape
        folder_path = os.path.join(output_dir, f"page_{page_num}")
        
        # Search for all and only .html files
        html_files = glob.glob(os.path.join(folder_path, "*.html"))
        
        # Iterates through the found files and processes each one
        for file_path in html_files:
            print("Processing:", file_path)
            # Scrape geodata
            restaurant_info = parser.extract_geo_info(file_path)
            restaurants_data.append(restaurant_info)

    return restaurants_data

# Translate regions from english to italian name
def translateRegions(regionColumn):
    translations = {
        "Aosta Valley": "Valle d'Aosta/Vallée d'Aoste",
        "Piedmont": "Piemonte",
        "Lombardy": "Lombardia",
        "Sicily": "Sicilia",
        "Tuscany": "Toscana",
        "Apulia": "Puglia",
        "Trentino-South Tyrol": "Trentino-Alto Adige/Südtirol",
        "Sardinia": "Sardegna"
    }
    regionColumn = regionColumn.replace(translations) 

    return regionColumn

def fetch_page(url):
    '''
    Take in input a url and try to send request to it
    '''
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP Error Status Code
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None

def parse_restaurant_urls(page_content):
    ''''
    Collect all the urls in the defined url
    '''
    soup = BeautifulSoup(page_content, "html.parser")
    restaurant_urls = []

    for link in soup.find_all("a", class_="link"):
        href = link.get("href")
        if href:
            restaurant_urls.append(f"https://guide.michelin.com{href}")

    return restaurant_urls

def scrape_michelin_restaurants(base_url, pages):
    all_restaurant_urls = []

    for page in range(1, pages+1):
        url = f"{base_url}/page/{page}"
        print(f"Fetching {url}...")
        page_content = fetch_page(url)

        if page_content:
            restaurant_urls = parse_restaurant_urls(page_content)
            all_restaurant_urls.extend(restaurant_urls)

    with open("michelin_restaurant_urls.txt", "w") as file:
        for restaurant_url in all_restaurant_urls:
            file.write(restaurant_url + "\n")

    print(f"Collected {len(all_restaurant_urls)} URL of restaurants.")

def download_HTML(url, folder_name):
    '''
    Function to download and save the HTML content of each restaurant in each page
    '''
    # Extract the resturant name from the URL
    match = re.search(r'[^/]+$', url)
    restaurant_name = match.group() if match else "unknown"

    # Do the request to the page
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the request has not been completed
        soup = BeautifulSoup(response.text, "html.parser")

        # Prettify the HTML content for readability
        html_content = soup.prettify()

        # Save the content in a .txt file in the specified folder
        file_path = os.path.join(folder_name, f"html_{restaurant_name}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(html_content)
        print(f"Saved HTML of {url} in {file_path}")

    except requests.RequestException as e:
        print(f"Errore nel scaricare {url}: {e}")

def process_urls(file_path):
    '''
    Function to divide the URL and handle the folders
    :param file_path:
    :return: None, create a file for each URL and a folder for each page
    '''
    # Read all the URL from the file
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Divide in group of 20(number of element in each page)
    for i in range(0, len(urls), 20):
        group_urls = urls[i:i + 20]

        # Create a folder for each group of 20 URLs
        folder_name = f"page_{i // 20 + 1}"
        os.makedirs(folder_name, exist_ok=True)

        # Download and save each URL in the respective group
        for url in group_urls:
            download_HTML(url, folder_name)

def extract_restaurant_info(file_path):
    '''
    Function to extract the restaurant info (restaurantName, city, postalCode, country, address,
    priceRange, cuisineType, description, facilitiesServices, creditCards, phoneNumber, website) from each restaurant HTML file
    :param file_path:
    :return: Dict of all the information of the respective restaurant collected for each column
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

        restaurant_info = {}

        # Find the Div with the principal information of the restaurant
        restaurantDetailsDiv = soup.find("div", class_="restaurant-details__components")
        mainInfo = restaurantDetailsDiv.select("div.data-sheet > div.row")

        # Extract this main information
        if mainInfo[0]:
            restaurant_info['restaurantName'] = mainInfo[0].find("h1", class_="data-sheet__title").text.strip()
        if mainInfo[1]:
            address_price = mainInfo[1].select("div.data-sheet__block > div.data-sheet__block--text")
            addressList = address_price[0].text.strip().split(",")

            restaurant_info['city'] = addressList[-3].strip()
            restaurant_info['postalCode'] = addressList[-2].strip()
            restaurant_info['country'] = addressList[-1].strip()
            restaurant_info['address'] = " ".join(addressList[:-3]).strip().replace("\n", "")

            restaurant_info['priceRange'], restaurant_info['cuisineType'] = address_price[1].text.strip().split("·")
            restaurant_info['priceRange'] = restaurant_info['priceRange'].strip()
            restaurant_info['cuisineType'] = restaurant_info['cuisineType'].strip()

        # Description
        restaurant_info['description'] = soup.find("div", class_="data-sheet__description").text.strip()

        # Facilities and Services
        facilities = soup.select("div.restaurant-details__services ul li")
        restaurant_info['facilitiesServices'] = [s.text.strip() for s in facilities if
                                                 s.text.strip()]  # Extract only non-empty items

        # Accepted Credit Cards - extract specific card names (Amex, Mastercard, Visa, etc.)
        credit_cards = soup.select("div.list--card img")
        restaurant_info['creditCards'] = [
            # Extracting the card name from the "data-src" attribute of the <img> tag.
            re.search(r"(?<=\/)([a-zA-Z]+)(?=-)", c.get("data-src"))[0].capitalize() for c in credit_cards if
            c.get("data-src")
        ]

        # Phone Number
        spansDetails = restaurantDetailsDiv.select(
            "section.section.section-main.section__text-componets.section__text-separator div.collapse__block-title div.d-flex span")
        restaurant_info['phoneNumber'] = spansDetails[0].text.strip()

        # Website URL
        website_link = soup.find("a", {"data-event": "CTA_website"})
        restaurant_info['website'] = website_link["href"].strip() if website_link else None

    return restaurant_info

def find_restaurants_updated(query_text, vocabulary_df, inverted_index, df):
    '''
    Find restaurants that match the given query using the inverted index
    Inputs:
    query: query string
    inverted_index: inverted index dictionary
    df: dataframe with restaurants data
    Outputs:
    restaurants_df: dataframe with restaurants that match the query
    '''
    # Preprocess the query
    query_tokens = engine.preprocessing(query_text)

    target_docs = []

    try:
        # Retrieve the term_ids for each token in the query
        term_ids = [vocabulary_df[vocabulary_df['term'] == token]['term_id'].iloc[0] for token in query_tokens]

        # Retrieve the document IDs for each term_id (from inverted index)
        # Create a list of sets containing document IDs for each term in the query
        doc_sets = [set(inverted_index[term_id]) for term_id in term_ids]

        # Find the common document IDs across all query terms
        common_docs = set.intersection(*doc_sets) if doc_sets else set()

        # If there are any common documents, add them to target_docs
        if common_docs:
            target_docs.extend(common_docs)

        # Convert target_docs to a list (if it's not already)
        target_docs = list(target_docs)

        # Retrieve the rows that match doc_ids in target_docs
        restaurants_df = df.loc[target_docs][:] # Consider now all the columns

        # Return the DataFrame with the matching restaurants
        return restaurants_df

    except Exception as e:
        print(f"Error in finding restaurants: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

def find_top_custom_restaurants(query, vocabulary_df, inverted_index, df, k=5):
    '''
    Find top-k restaurants that match the given query using the inverted index and apply scoring.
    '''
    # Collect all the info inserted in the UI
    query_text = query.get('description', '')
    cuisine_type = query.get('cuisineType', '')
    facilities = query.get('facilitiesServices', [])
    price_preference = query.get('priceRange', '')
    num_results = query.get('num_results', 0)
    if not isinstance(num_results, int) or num_results <= 0:
        num_results = k
    # Preprocess the query and find the meaningful documents based on the description
    matching_docs_df = find_restaurants_updated(query_text, vocabulary_df, inverted_index, df)

    if matching_docs_df.empty:
        return pd.DataFrame()

    # Compute the total number of documents
    n_docs = len(df)

    # Preprocess the query and get the unique terms (tokens)
    processed_query = engine.preprocessing(query_text)
    query_tokens = list(set(processed_query))

    # Compute the TF-IDF vector of the give query
    query_tf_idf_scores = {}
    for term in query_tokens:
        if term in vocabulary_df['term'].values:
            term_id = vocabulary_df[vocabulary_df['term'] == term].index[0]
            n_term_docs = len(inverted_index[term_id])  # Number of documents in which is present the current term
            IDF = np.log10(n_docs / n_term_docs)
            TF = processed_query.count(term) / len(processed_query)
            query_tf_idf_scores[term] = TF * IDF
            #print(f'Debug: {TF} * {IDF}')
    query_norm = np.linalg.norm(list(query_tf_idf_scores.values()))  # Compute the query norm

    # Compute the TF-IDF vectors for each document in matching_docs_df
    doc_vectors = []
    for _, row in matching_docs_df.iterrows():
        doc_vector = {}
        doc_tokens = row['description'].split()  # Tokenize the description of the document
        for term in set(doc_tokens):
            if term in vocabulary_df['term'].values:
                term_id = vocabulary_df[vocabulary_df['term'] == term].index[0]
                n_term_docs = len(inverted_index[term_id])
                IDF = np.log10(n_docs / n_term_docs)
                TF = doc_tokens.count(term) / len(doc_tokens)
                doc_vector[term] = TF * IDF
        doc_vectors.append(doc_vector)

    # Compute the norms of each document vector
    doc_norms = [np.linalg.norm(list(doc_vector.values())) for doc_vector in doc_vectors]

    # Define the weights
    description_weight = 0.1
    cuisine_weight = 0.3
    facilities_weight = 0.3
    price_weight = 0.3

    scored_restaurants = []

    for i, (idx, row) in enumerate(matching_docs_df.iterrows()):
        score = 0
        # Description Match - Compute the cosine similarity between the query and the document
        description_score = 0
        doc_vector = doc_vectors[i]

        # Compute the cosine similarity for each common term
        common_terms = set(doc_vector.keys()).intersection(query_tf_idf_scores.keys())
        for term in common_terms:
            description_score += doc_vector[term] * query_tf_idf_scores[term]

        # Normalize the cosine similarity
        if doc_norms[i] * query_norm != 0:
            description_score /= (doc_norms[i] * query_norm)

        score += description_score * description_weight

        # CuisineType Match - if the cuisine type is matched
        # then we add to the custom score the value of the respective weight
        cuisine_score = 1 if cuisine_type and row['cuisineType'] == cuisine_type else 0
        score += cuisine_score * cuisine_weight

        # Facilities Match - for each facility & service matched we sum 1 then we take the value of total matches and
        # we divide it by the total number of the facilities and servicies matched,
        # at the end we add the value of the weight times this value just computed
        facilities_score = 0
        restaurant_facilities = row['facilitiesServices']
        if isinstance(restaurant_facilities, str):
            restaurant_facilities = eval(restaurant_facilities)

        facilities_matches = sum([1 for facility in facilities if facility in restaurant_facilities])
        facilities_score = facilities_matches / len(facilities) if facilities else 0
        score += facilities_score * facilities_weight

        # Price Range Match - if the price range is matched
        # then we add to the custom score the value of the respective weight
        price_score = 1 if price_preference and row['priceRange'] == price_preference else 0
        score += price_score * price_weight

        scored_restaurants.append((score, row))

    # Find the top-k restaurants
    top_k_restaurants = heapq.nlargest(num_results, scored_restaurants, key=lambda x: x[0])
    # Convert the result in a pandas DataFrame
    if top_k_restaurants:
        top_k_df = pd.DataFrame([restaurant[1] for restaurant in top_k_restaurants])
        top_k_df['customScore'] = [restaurant[0] for restaurant in top_k_restaurants]
        # Save the result in a different .tsv file for represent those in the map plot
        top_k_df.to_csv('top_k_result.tsv', sep='\t', index=False)
        return top_k_df[['restaurantName', 'address', 'description', 'website', 'customScore']]
    else:
        return pd.DataFrame()
