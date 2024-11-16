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
import aiohttp
import asyncio
import aiofiles

# Point 1.1
def scrape_restaurant_links(base_url):
    print("I'm starting to Scrape!")
    page = 1
    # Link list to return as a file.txt
    all_links = []

    while True:
        # Build URL of current page which is base_url + # of the current page
        url = f"{base_url}{page}"
        print(f"Fetching {url}...")
        response = requests.get(url)

        # Check if the HTTP request was successful, otherwise stop scraping and print error message
        if response.status_code != 200:
            print(f"Errore nel caricamento della pagina {page}")
            break

        # Parsing of the HTML page using Beautiful Soup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Dig in HTML structure to search every restaurnt URL in the current page, first element is the <div> tag that has following class
        for class1_div in soup.select("div.card__menu-content.card__menu-content--flex.js-match-height-content"):
            # Next is the <h3> tag that has the following class, this tag also has an attribute "href" 
            # with the link of the current restaurant, we need to retrive this link 
            h3 = class1_div.select_one("h3.card__menu-content--title.pl-text.pl-big.js-match-height-title a")
            if h3:
                link = h3.get("href")
                # Build the link and add to link list
                full_link = "https://guide.michelin.com" + link if link else None
                if full_link:
                    all_links.append(full_link)


        # In this section of the scraping algorithm, we search for the "next page" button so that
        # we can scrape the other pages links.

        # First step is to find the div that contains pages button which is the div with the followind class
        pagination_lis = soup.select("div.js-restaurant__bottom-pagination ul li")

        # Next, we should find the button which has "active" in the class (which indicate the current page)
        active_index = None
        for i, li in enumerate(pagination_lis):
            if li.select_one("a.active"):
                active_index = i
                break

        # After we have found the current button, we search for immediate next button, 
        # who contains link to the next page in the <a> tag, if next page is present.
        if active_index is not None and active_index + 1 < len(pagination_lis):
            next_page = pagination_lis[active_index + 1].select_one("a")
            if next_page and next_page.get("href"):
                page += 1
            else:
                break
        else:
            break

    # Save every URL in a .txt file
    with open("michelin_restaurant_urls.txt", "w") as file:
        for link in all_links:
            file.write(link + "\n")

    print(f"Scraping completed. {len(all_links)} link saved in michelin_restaurant_urls.txt.")

# Point 1.2

# Load URLs from a file
async def load_urls(file_path):
    async with aiofiles.open(file_path, 'r') as f:
        urls = [line.strip() for line in await f.readlines()]
    return urls

# Download a single URL
async def download_url(session, url, folder_name):
    # HTTP headers to mimic a browser request and avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
        'Referer': 'https://guide.michelin.com/',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                content = await response.text()
                
                # Extract the restaurant name from the URL so that the HTML could be saved using this name
                match = re.search(r'[^/]+$', url)
                restaurant_name = match.group() if match else "unknown"
                filename = os.path.join(folder_name, f"{restaurant_name}.html")

                # Asynchronously save the content of HTML page to a file
                async with aiofiles.open(filename, 'w') as f:
                    await f.write(content)
                print(f"Downloaded: {url}")
            else:
                # Log the failure if the response status is not 200
                print(f"Failed to download {url}: Status {response.status}")
    except Exception as e:
        # Handle and log any other exceptions during the download process
        print(f"Error downloading {url}: {e}")

# Download all URLs, organizing them into folders
async def download_all(urls, output_dir):
    CONCURRENT_REQUESTS = 20  # Limit the number of concurrent requests

    # Set up a connector to limit the number of concurrent connections
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    # Create an HTTP session
    async with aiohttp.ClientSession(connector=connector) as session:
        folder_index = 1

        # Process 20 URLs at a time and save them in different folders
        for i in range(0, len(urls), 20):  
            folder_name = os.path.join(output_dir, f"page_{folder_index}")
            os.makedirs(folder_name, exist_ok=True)
            tasks = [download_url(session, url, folder_name) for url in urls[i:i+20]]
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
            folder_index += 1

# Point 1.3

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
            restaurant_info['cuisineType'] = restaurant_info['cuisineType'].strip().split(",")

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

# Point 4.1

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

def iterate_geo_folders(output_dir):
    # List to store restaurant data
    restaurants_data = []

    # Loop through all files in the directory and extract information
    # Itera attraverso le pagine da 1 a 100
    for page_num in range(1, 101):
        # Costruisci il percorso della cartella per ciascuna pagina
        folder_path = os.path.join(output_dir, f"page_{page_num}")
        
        # Trova tutti i file di testo che iniziano con "html_" nella cartella corrente
        html_files = glob.glob(os.path.join(folder_path, "html_*.txt"))
        
        # Itera attraverso i file trovati e processa ciascuno
        for file_path in html_files:
            print(file_path)
            restaurant_info = extract_geo_info(file_path)
            restaurants_data.append(restaurant_info)

    return restaurants_data

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
    try:
        response = requests.get(url)
        response.raise_for_status()  # Controlla se ci sono errori HTTP
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta: {e}")
        return None

def parse_restaurant_urls(page_content):
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


# Function to download and save the HTML content of each page
def download_HTML(url, folder_name):
    # Extract the resturant name from the URL
    match = re.search(r'[^/]+$', url)
    restaurant_name = match.group() if match else "unknown"

    # Do the request to the page
    try:
        response = requests.get(url)
        response.raise_for_status()  # Eaise an exception if the request has not been completed
        soup = BeautifulSoup(response.text, "html.parser")

        # Prettify the HTML content for readibility
        html_content = soup.prettify()

        # Save the content in a .txt file in the specified folder
        file_path = os.path.join(folder_name, f"html_{restaurant_name}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(html_content)
        print(f"Saved HTML of {url} in {file_path}")

    except requests.RequestException as e:
        print(f"Errore nel scaricare {url}: {e}")

# Function to divide the URL and handle the folders
def process_urls(file_path):
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

# Funzione per estrarre le informazioni del ristorante dall'HTML
def extract_restaurant_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

        restaurant_info = {}

        # Div con le informazioni principali del ristorante
        restaurantDetailsDiv = soup.find("div", class_="restaurant-details__components")
        mainInfo = restaurantDetailsDiv.select("div.data-sheet > div.row")

        # Estrazione delle informazioni principali (nome, indirizzo, ecc.)
        if mainInfo[0]:
            restaurant_info['restaurantName'] = mainInfo[0].find("h1", class_="data-sheet__title").text.strip()
        if mainInfo[1]:
            indirizzo_price = mainInfo[1].select("div.data-sheet__block > div.data-sheet__block--text")
            indirizzoList = indirizzo_price[0].text.strip().split(",")

            restaurant_info['city'] = indirizzoList[-3].strip()
            restaurant_info['postalCode'] = indirizzoList[-2].strip()
            restaurant_info['country'] = indirizzoList[-1].strip()
            restaurant_info['address'] = " ".join(indirizzoList[:-3]).strip().replace("\n", "")

            restaurant_info['priceRange'], restaurant_info['cuisineType'] = indirizzo_price[1].text.strip().split("·")
            restaurant_info['priceRange'] = restaurant_info['priceRange'].strip()
            restaurant_info['cuisineType'] = restaurant_info['cuisineType'].strip()

        # Descrizione
        restaurant_info['description'] = soup.find("div", class_="data-sheet__description").text.strip()

        # Servizi e strutture
        facilities = soup.select("div.restaurant-details__services ul li")
        restaurant_info['facilitiesServices'] = [s.text.strip() for s in facilities if
                                                 s.text.strip()]  # Extract only non-empty items

        # Carte di credito accettate - extract specific card names (Amex, Mastercard, Visa, etc.)
        credit_cards = soup.select("div.list--card img")
        restaurant_info['creditCards'] = [
            # Extracting the card name from the "data-src" attribute of the <img> tag.
            re.search(r"(?<=\/)([a-zA-Z]+)(?=-)", c.get("data-src"))[0].capitalize() for c in credit_cards if
            c.get("data-src")
        ]

        # Numero di telefono
        spansDetails = restaurantDetailsDiv.select(
            "section.section.section-main.section__text-componets.section__text-separator div.collapse__block-title div.d-flex span")
        restaurant_info['phoneNumber'] = spansDetails[0].text.strip()

        # URL sito web
        website_link = soup.find("a", {"data-event": "CTA_website"})
        restaurant_info['website'] = website_link["href"].strip() if website_link else None

    return restaurant_info

def preprocessing(doc):
    '''
    Function that preprocesses a document
    Input:
    doc: document to preprocess
    Output:
    tokens: list of cleaned tokens
    '''
    # Tokenize the document
    tokens = word_tokenize(doc)

    # Turn all words to lowercase
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    stops = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stops]

    # Remove puntuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # tokens = [re.sub(r'[^\w\s]','',token) for token in tokens]

    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Handle possessivenes
    def handle_possessives(token):
      if token.endswith("'s"):
        return token[:-2]  # Remove the "'s" part
      return token
    tokens = [handle_possessives(token) for token in tokens]

    # Normalize tokens
    tokens = [unidecode(token) for token in tokens]

    # Remove apostrophes
    tokens = [token.replace("'"," ").replace("-"," ") for token in tokens]

    # Remove numbers and empty strings
    tokens = [token for token in tokens if token != "" and not token.isdigit()]

    # Now split any token that contains a space into separate words
    final_tokens = []
    for token in tokens:
        # If the token contains spaces, split it into individual words
        if " " in token:
            final_tokens.extend(token.split())  # Extend adds each word separately to the list
        else:
            final_tokens.append(token)

    return final_tokens

def find_restaurants(query, vocabulary_df, inverted_index, df):
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
    query_tokens = preprocessing(query)

    target_docs = []

    try:
      # Retrieve the term_ids for each token in the query
      term_ids = [vocabulary_df[vocabulary_df['term'] == token]['term_id'].iloc[0] for token in query_tokens]

      # Retrieve the document IDs for each term_id (from inverted index)
      # Create a list of sets containing document IDs for each term in the query
      doc_sets = [set(inverted_index[term_id]) for term_id in term_ids]

      # Find the common document IDs across all query terms
      common_docs = set.intersection(*doc_sets)

      # If there are any common documents, add them to target_docs
      if common_docs:
          target_docs.extend(common_docs)

      # Convert target_docs to a list (if it's not already)
      target_docs = list(target_docs)

      # Retrieve the rows that match doc_ids in target_docs
      restaurants_df = df.loc[target_docs][['restaurantName', 'address', 'description', 'website']]

      # Return the DataFrame with the matching restaurants
      return restaurants_df

    except:
      print("No restaurants found for the given query.")

def tf_idf(term_id, inverted_index, preprocessed_docs, vocabulary_df, n):
  '''
  Calculate the TF-IDF scores for a given term
  Inputs:
  term_id: term id
  inverted_index: dictionary storing the documents that each term appears in
  preprocessed_docs: dictionary storing all the preprocessed documents
  vocabulary_df: dataframe containing the vocabulary of terms
  n = total number of documents
  Output:
  tf_idf_scores: vector of TF-IDF scores for the given term
  '''

  term = vocabulary_df['term'][term_id] # get term from term_id
  n_term = len(inverted_index[term_id]) # number of documents that contain the term
  #print(n_term)
  IDF = np.log10(n / n_term) # calculate IDF of the term, inverse document frequency
  #print(f"IDF= {IDF}")
  tf_idf_scores = [] # initialize list to store TF-IDF scores

  for doc_id in inverted_index[term_id]:
    #TF = preprocessed_docs[doc_id].count(term) / len(preprocessed_docs[doc_id]) # RELATIVE term frequency (tf)
    TF = preprocessed_docs[doc_id].count(term) # raw term frequency (tf)
    #print(f"TF = {TF}")
    tf_idf_scores.append(TF * IDF) # calculate TF-IDF score

  return tf_idf_scores

def top_k_restaurants(query, inverted_index, vocabulary_dict, doc_tf_idf_scores, df, k=5, n = 0):
  '''
  Find the top k restaurants that match the given query using the TF-IDF scores
  Inputs:
  query: query string
  inverted_index: inverted index dictionary
  vocabulary_dict: dictionary containing the vocabulary of terms and their indeces
  doc_tf_idf_scores: dictionary storing the TF-IDF scores for each term in each document
  df: dataframe with restaurants data
  k: number of restaurants to return
  Outputs:
  restaurants_df: dataframe with restaurants that match the query
  '''
  processed_query = preprocessing(query) # processed query
  query_tokens = list(set(processed_query)) # unique query tokens
  # print(query_tokens) # debugging line
  # Find all docs to consider
  docs_to_consider = [] # initialize list to store documents to consider (non-zero intersection with the query tokens)

  for token in query_tokens:
    if vocabulary_dict[token]: # check if the token is in the vocabulary
      token_id = vocabulary_dict[token] # get the term_id of the token
      docs_to_consider.extend(inverted_index[token_id]) # add the documents that contain the token to the docs to consider

  docs_to_consider = list(set(docs_to_consider)) # remove duplicates
  # Calculate the TF-IDF score of the query
  query_tf_idf_scores = [] # initialize list to store the TF-IDF scores of the query
  for term in query_tokens:
    term_id = vocabulary_dict[term] # get the term_id of the term
    #print(inverted_index[term_id]) # debugging line
    n_term = len(inverted_index[term_id]) # number of documents that contain the term
    IDF = np.log10(n / n_term) # calculate IDF of the term
    TF = processed_query.count(term) # calculate TF of the term
    #print(f"TF = {TF}") # debugging line
    #print(f"IDF = {IDF}") # debuggin line
    query_tf_idf_scores.append((term_id, TF * IDF)) # calculate TF-IDF score

  query_tf_idf_scores.sort(key=lambda x: x[0]) # sort the query_tf_idf_scores in order of term_id

  query_norm = np.linalg.norm(np.array([score for _, score in query_tf_idf_scores])) # calculate the norm of the query
  #print(f"query tf_idf_scores: {query_tf_idf_scores}") # debuggin line
  #print(f"query norm: {query_norm}") # debugging line
  # calculate document norms
  doc_norms = {doc_id: np.linalg.norm(np.array([doc_tf_idf_scores[doc_id][i][1] for i in range(len(doc_tf_idf_scores[doc_id]))])) for doc_id in docs_to_consider}

  # Function that returns two lists of tuples (term, query_tf_idf) and (term, doc_tf_idf) such that
  # the terms are in the intersection of the query terms and the doc's terms
  def query_doc_intersection(query_tf_idf_scores, doc_tf_idf_scores):
    '''
    Calculate the intersection of the query and the document
    Inputs:
    query_terms: list of sorted unique query terms
    doc_terms: list of sorted unique document terms
    Output:
    query_intersection: list of tuples (term, query_tf_idf)
    doc_intersection: list of tuples (term, doc_tf_idf)
    '''
    query_intersection = [] # initialize list to store (term, query_tf_idf) tuples in the intersection
    doc_intersection = [] # initialize list to store (term, doc_tf_idf) tuples in the intersection
    i, j = 0, 0 # initialize two pointers
    while i<len(query_tf_idf_scores) and j<len(doc_tf_idf_scores):
      if query_tf_idf_scores[i][0] == doc_tf_idf_scores[j][0]:
        query_intersection.append(query_tf_idf_scores[i])
        doc_intersection.append(doc_tf_idf_scores[j])
        i += 1
        j += 1
      elif query_tf_idf_scores[i][0] < doc_tf_idf_scores[j][0]:
        i += 1
      else:
        j += 1
    return query_intersection, doc_intersection

  # Calculate cosine-similarity between the query and each document
  cosine_similarity = defaultdict(float) # initialize dictionary to store the cosine similarity results
  for doc_id in docs_to_consider:
    query_intersection, doc_intersection = query_doc_intersection(query_tf_idf_scores, doc_tf_idf_scores[doc_id]) # find the
    cosine_similarity[doc_id] = np.dot(np.array([score for _, score in query_intersection]), np.array([score for _, score in doc_intersection])) / (query_norm * doc_norms[doc_id])

  # Sort the cosine similarities in descending order
  sorted_cosine_similarity = sorted(cosine_similarity.items(), key=lambda x: x[1], reverse=True) # list of tuples

  # Get the top k restaurants
  top_k_restaurants = sorted_cosine_similarity[:min(k,len(sorted_cosine_similarity))]

  '''
  Information to store in the end data frame:
  restaurantName
  address
  description
  website
  Similarity score (between 0 and 1)
  '''

  top_k_restaurant_idx = [doc_id for doc_id, _ in top_k_restaurants]
  top_k_restaurant_scores = [score for _, score in top_k_restaurants]
  #print([score for _, score in top_k_restaurants])

  # build result dataframe
  restaurants_df = df.loc[top_k_restaurant_idx][['restaurantName', 'address', 'description', 'website']]
  restaurants_df['Similarity score'] = top_k_restaurant_scores

  return restaurants_df

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
    query_tokens = preprocessing(query_text)

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
        restaurants_df = df.loc[target_docs][
            ['restaurantName', 'address', 'description', 'website', 'cuisineType', 'facilitiesServices', 'priceRange']]

        # Return the DataFrame with the matching restaurants
        return restaurants_df

    except Exception as e:
        print(f"Error in finding restaurants: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


def find_top_custom_restaurants(query, vocabulary_df, inverted_index, df, k=5):
    '''
    Find top-k restaurants that match the given query using the inverted index and apply scoring.
    '''
    # Recupera i valori di query per i criteri specifici
    query_text = query.get('description', '')
    cuisine_type = query.get('cuisineType', '')
    facilities = query.get('facilitiesServices', [])
    price_preference = query.get('priceRange', '')
    num_results = query.get('num_results', 0)
    if not isinstance(num_results, int) or num_results <= 0:
        num_results = k
    # Step 1: Preprocessa la query e trova i documenti pertinenti
    matching_docs_df = find_restaurants_updated(query_text, vocabulary_df, inverted_index, df)

    if matching_docs_df.empty:
        return pd.DataFrame()

    # Calcola il numero totale di documenti per l'IDF
    n_docs = len(df)

    # Preprocessa la query e ottieni i termini unici
    processed_query = preprocessing(query_text)
    query_tokens = list(set(processed_query))

    # Step 2: Calcola il vettore TF-IDF della query
    query_tf_idf_scores = {}
    for term in query_tokens:
        if term in vocabulary_df['term'].values:
            term_id = vocabulary_df[vocabulary_df['term'] == term].index[0]
            n_term_docs = len(inverted_index[term_id])  # Numero di documenti contenenti il termine
            IDF = np.log10(n_docs / n_term_docs)
            TF = processed_query.count(term) / len(processed_query)
            query_tf_idf_scores[term] = TF * IDF
            #print(f'Debug: {TF} * {IDF}')
    query_norm = np.linalg.norm(list(query_tf_idf_scores.values()))  # Calcola la norma della query

    # Step 3: Calcola i vettori TF-IDF per i documenti in matching_docs_df
    doc_vectors = []
    for _, row in matching_docs_df.iterrows():
        doc_vector = {}
        doc_tokens = row['description'].split()  # Tokenizza la descrizione del documento
        for term in set(doc_tokens):
            if term in vocabulary_df['term'].values:
                term_id = vocabulary_df[vocabulary_df['term'] == term].index[0]
                n_term_docs = len(inverted_index[term_id])
                IDF = np.log10(n_docs / n_term_docs)
                TF = doc_tokens.count(term) / len(doc_tokens)
                doc_vector[term] = TF * IDF
        doc_vectors.append(doc_vector)

    # Calcola le norme dei vettori documenti per la normalizzazione
    doc_norms = [np.linalg.norm(list(doc_vector.values())) for doc_vector in doc_vectors]

    # Definizione dei pesi
    description_weight = 0.1
    cuisine_weight = 0.3
    facilities_weight = 0.3
    price_weight = 0.3

    scored_restaurants = []

    for i, (idx, row) in enumerate(matching_docs_df.iterrows()):
        score = 0

        # (a) Description Match - Calcola la similarità coseno tra la query e il documento
        description_score = 0
        doc_vector = doc_vectors[i]

        # Calcola la similarità coseno per i termini in comune
        common_terms = set(doc_vector.keys()).intersection(query_tf_idf_scores.keys())
        for term in common_terms:
            description_score += doc_vector[term] * query_tf_idf_scores[term]

        # Normalizza la similarità coseno
        if doc_norms[i] * query_norm != 0:
            description_score /= (doc_norms[i] * query_norm)

        score += description_score * description_weight

        # (b) Cuisine Match
        cuisine_score = 1 if cuisine_type and row['cuisineType'] == cuisine_type else 0
        score += cuisine_score * cuisine_weight

        # (c) Facilities Match
        facilities_score = 0
        restaurant_facilities = row['facilitiesServices']
        if isinstance(restaurant_facilities, str):
            restaurant_facilities = eval(restaurant_facilities)

        facilities_matches = sum([1 for facility in facilities if facility in restaurant_facilities])
        facilities_score = facilities_matches / len(facilities) if facilities else 0
        score += facilities_score * facilities_weight

        # (d) Price Range Match
        price_score = 1 if price_preference and row['priceRange'] == price_preference else 0
        score += price_score * price_weight

        scored_restaurants.append((score, row))

    # Step 5: Trova i top-k ristoranti
    top_k_restaurants = heapq.nlargest(num_results, scored_restaurants, key=lambda x: x[0])
    # Converte il risultato in un DataFrame
    if top_k_restaurants:
        top_k_df = pd.DataFrame([restaurant[1] for restaurant in top_k_restaurants])
        top_k_df['customScore'] = [restaurant[0] for restaurant in top_k_restaurants]
        return top_k_df[['restaurantName', 'address', 'description', 'website', 'customScore']]
    else:
        return pd.DataFrame()

def query_doc_intersection_enanched(query_tf_idf_scores, doc_tf_idf_scores):
    '''
    Calculate the intersection of the query and the document
    Inputs:
    query_terms: list of sorted unique query terms
    doc_terms: list of sorted unique document terms
    Output:
    query_intersection: list of tuples (term, query_tf_idf)
    doc_intersection: list of tuples (term, doc_tf_idf)
    '''
    query_intersection = [] # initialize list to store (term, query_tf_idf) tuples in the intersection
    doc_intersection = [] # initialize list to store (term, doc_tf_idf) tuples in the intersection
    i, j = 0, 0 # initialize two pointers
    while i<len(query_tf_idf_scores) and j<len(doc_tf_idf_scores):
      if query_tf_idf_scores[i][0] == doc_tf_idf_scores[j][0]:
        query_intersection.append(query_tf_idf_scores[i])
        doc_intersection.append(doc_tf_idf_scores[j])
        i += 1
        j += 1
      elif query_tf_idf_scores[i][0] < doc_tf_idf_scores[j][0]:
        i += 1
      else:
        j += 1
    return query_intersection, doc_intersection
