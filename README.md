# ADM - Homework 3: Michelin Restaurants in Italy, Group #14

This GitHub repository contains the implementation of the third homework of the **Algorithmic Methods of Data Mining** course for the master's degree in Data Science at Sapienza (2024-2025). This homework was completed by Group #14 in the academic year 2024–2025. The details of the assignement are specified here: https://github.com/Sapienza-University-Rome/ADM/tree/master/2024/Homework_3

**Team Members:**
* Xavier Del Giudice, 1967219, delgiudice.1967219@studenti.uniroma1.it
* Alessio Iacono, 1870276, iacono.1870276@studenti.uniroma1.it
* Géraldine Valérie Maurer, 1996887, gmaurer08@gmail.com

The ```main.ipynb``` with the main script can be visualized in the jupyter notebook viewer: **INSERT LINK TO NBVIEWER**

## Repository Structure

```
├── GeoJson/                     # Directory containing files for map visualization
│   ├── italy_map+geojson.json       # GeoJSON file with Italian map data for visualization
│   └── limits_IT_regions.geojson    # GeoJSON file with Italian regional boundaries
├── functions/                   # Directory containing core project modules
│   ├── crawler.py               # Module for scraping Michelin restaurant data
│   ├── parser.py                # Module for parsing and extracting data from HTML files
│   ├── engine.py                # Implementation of the search engine (conjunctive and ranked search)
│   ├── search_restaurants_ui.py # User interface for advanced search and custom scoring
│   └── utils.py                 # Helper functions for text preprocessing, scoring, and utilities
├── main.ipynb                   # Main notebook containg all the runned cells
├── .gitignore                   # File to specify files and directories ignored by Git
├── README.md                    # Project documentation
└── LICENSE                      # License file for the project
```

Here are links to all the files:

* [GeoJson](GeoJson): Directory containing files for map visualization
  * [italy_map+geojson.json](GeoJson/italy_map+geojson.json): GeoJSON file with Italian map data for visualization
  * [limits_IT_regions.geojson](GeoJson/limits_IT_regions.geojson): GeoJSON file with Italian regional boundaries
* [functions](functions): Directory containing core project modules
  * [crawler.py](functions/crawler.py): Module for scraping Michelin restaurant data
  * [parser.py](functions/parser.py): Module for parsing and extracting data from HTML files
  * [engine.py](functions/engine.py): Implementation of the search engine (conjunctive and ranked search)
  * [search_restaurants_ui.py](functions/search_restaurants_ui.py): User interface for advanced search and custom scoring
  * [limits_IT_regions.geojson](functions/limits_IT_regions.geojson): Helper functions for text preprocessing, scoring, and utilities
* [main.ipynb](main.ipynb): Main notebook containg all the runned cells
* [.gitignore](.gitignore): File to specify files and directories ignored by Git
* [README.md](README.md): Project documentation
* LICENSE: License file for the project

---

Here’s the **Table of Contents** with hyperlinks for easy navigation:

---

# Table of Contents

1. [Project Overview](##project-overview)  
2. [Data Collection](#point-1-data-collection)  
   - [Technologies Used](#technologies-used)  
   - [Datasets](#datasets)  
   - [Install Dependencies](#install-dependencies)  
   - [Steps of the Script](#steps-of-the-script)  
      1. [Get the List of Michelin Restaurants](#get-the-list-of-michelin-restaurants)  
      2. [Crawl Michelin Restaurant Pages](#crawl-michelin-restaurant-pages)  
      3. [Parse Downloaded Pages](#parse-downloaded-pages)  
3. [Search Engine](#point-2-search-engine)  
   - [Libraries Used](#libraries-used)  
   - [Steps](#steps)  
   - [Functions in `engine.py`](#functions-in-enginepy-used-in-point-2)  
   - [How to Use](#how-to-use)  
4. [Define a New Score!](#point-3-define-a-new-score)  
5. [Interactive Map of Italian Regions](#point-4-interactive-map-of-italian-regions-with-restaurants-by-price-range-and-top-k-restaurants)  
   - [Features](#features)  
   - [Technical Implementation](#technical-implementation)  
   - [How to Run the Application](#how-to-run-the-application)  
6. [Advanced Search Engine](#point-5-advanced-search-engine)  
7. [License](#license)  

---

This structure will allow users to click and jump directly to specific sections in your README file. Let me know if there's anything else you'd like to adjust!

---
## Project Overview

## 1. Data collection

This project require to scrape data from [Italian Michelin guide website](https://guide.michelin.com/en/it/restaurants), this process is divided into various steps:

### Technical Implementation

- **Technologies Used:**
  - *requests*: Library for HTTP request
  - *bs4* (BeautifulSoup): Parse HTML and extract information
  - *aiohttp*: Library for asynchronous HTTP requests
  - *asyncio*: Library for asynchronous programming
  - *aiofiles*: Library for asynchronous file I/O
  - *os*: Library for interacting with the operating system
  - *re*: Library for regular expressions, in particual, used to retrive credit card names.

- **Datasets:**
  - At the end of the process, the **restaurant dataset**, that contains information such as restaurant name, city, address, and cuisine types, will be created and stored in a .tsv file (*restaurants_data.tsv*).

### Install dependencies:

Before starting the script, these we install each dependency needed:

```bash
   %pip install unidecode geopy plotly dash aiofiles aiohttp nltk ipywidgets requests bs4 pandas
```

### Steps of the script:

1. #### **Get the list of Michelin restaurants**
   Our script loops through multiple pages, extracting links to individual restaurant pages from the HTML structure using **requests** and **BeautifulSoup**. 
   In particular, to scroll the pages, we look for the button of the active page (button with "active" in its class) and retrieve the link contained in the immediately following button.
   It stops when there are no more pages to scrape or if an error occurs.

   The collected links are saved in a file called *soupUrls.txt*

2. #### **Crawl Michelin restaurant pages** (IMPROVED VERSION)

   Our script downloads HTML content from a list of URLs stored in a file (*soupUrls.txt*) using **asynchronous** and **concurrent** programming with aiohttp, asyncio, and aiofiles to speed up the process. <br />

   It performs the following steps:

   -  **Load URLs**: Reads URLs asynchronously from *soupUrls.txt* file.
   -  **Download Content**: For each URL, sends an HTTP GET request with headers mimicking a browser, saves the response as an HTML file, and names the files uniquely using a hash of the URL to avoid duplicate files.
   -  **Concurrency Management**: Uses asynchronous tasks and a connection limit to efficiently download multiple pages concurrently without overloading the server.
   - **Output Management**: Creates a directory (downloads) to store the downloaded HTML files.

3. #### Parse downloaded pages

   Our script extracts detailed information about restaurants from HTML files downloaded earlier and saves the data into a structured tabular format (TSV file). 
   
   Here's a brief overview:

   -  **Input**: Reads HTML files from the downloads directory.
   -  **Parsing**: Uses BeautifulSoup to parse HTML and extract information needed, at the end, data will be stored in dictionary.
   -  **Output**: Aggregates the restaurant dictionaries into a Pandas DataFrame and saves the data as a TSV file (*restaurants_data.tsv*).

## 2. Search Engine

In this part of the project we implement two search engines:
* **Conjunctive Search Engine**: returns only the restaurants where all query terms are present in the associated description
* **Ranked Search Engine**: returns the top-k restaurants sorted by similarity to the query, using TF-IDF scores and Cosine Similarity

### Libraries Used:
* nltk (preprocessing)
* pandas (create vocabulary file)
* numpy (cosine-similarity)
* ipywidgets (interactive search engine)
* IPython.display (display widgets)
* defaultdict (store inverted_index, updated_inverted_index, preprocessed_docs and others)
* pickle (store inverted_index in pickle file)
* unidecode (normalize tokens)

### Steps:
1. Preprocessing: remove stopwords, punctuation, apply stemming, and perform other useful data cleaning steps
2. Create a vocabulary.csv file that associates terms from the restaurant descriptions to unique term IDs
3. Build inverted index that maps unique term IDs to document IDs where the term appears in
4. Implement the conjunctive search engine that processes input query terms and returns only the restaurants that contain all the query terms in their description

![search2 1](https://github.com/user-attachments/assets/c2eea2a1-bb0a-48e9-bbdb-a5cee7390264)

6. Calculate the TF-IDF scores for each token contained in the restaurant's description
7. Build an updated inverted index that maps unique term IDs with a list of tuples (doc ID, TF-IDF)
8. Implement the ranked search engine that processes input query terms and uses cosine-similarity with TF-IDF scores to return the top k restaurants

![search2 2](https://github.com/user-attachments/assets/74a89d65-7957-4bc2-991b-39e6d3cc6884)

### Functions in engine.py used in Point 2
* ```preprocessing```: preprocesses documents and query
* ```find_restaurants```: implements first search engine
* ```tf_idf```: computes the TF-IDF scores for a given term and a list of preprocessed documents
* ```top_k_restaurants```: implements second search engine

### How to use:
* Running the cells associated with the first and second search engine triggers an interactive search bar where the user can input a query
* Click on 'Search', and the ```engine.py``` will be called to execute the appropriate search engine and return the matching restaurants
   
## 3. Define a New Score!

https://github.com/user-attachments/assets/b6d04672-8e3c-4f37-b4bf-ee574cd8034f

## 4. Interactive Map of Italian Regions with restaurants by price range and Top-k restaurants

This project uses Dash and Plotly to create an interactive map of Italian regions. Users can select a region by clicking on the map and view additional details, such as the restaurants in the selected region.

### Features

1. **Visualization of the Map of Italy:**
   - Displays all Italian regions.
   - Each region can be selected by clicking on the map.

2. **Detailed Map of the Selected Region:**
   - Once a region is selected, a detailed map of the region is displayed.
   - Restaurants in the region appear as markers on the map, each with a different color for better viewing.

3. **Interactive Hover:**
   - Hovering over restaurant markers displays detailed information, including:
     - Restaurant name.
     - City.
     - Address.
     - Cuisine types.

4. **Responsive Layout:**
   - The app dynamically adjusts the layout to fit different screen sizes, with both maps displayed side by side, each taking up half the screen width.

### Technical Implementation

- **Technologies Used:**
  - Dash: For building the web application.
  - Plotly: For creating interactive maps.
  - GeoPandas: For handling geospatial data and processing the Italian region boundaries (GeoJSON).
  - Pandas: For data manipulation of the restaurant dataset.

- **Datasets:**
  - The restaurant dataset contains information such as restaurant name, city, address, and cuisine types, along with their geographical coordinates (latitude and longitude).
  - geodata dataset contains region and coordinates information about each restaurant.

### How to Run the Application

1. **Install Dependencies:**
   ```bash
   pip install dash plotly geopandas pandas

2. **Steps to take**
    1. Run Point 3 to create vocabulary.tsv and top_k_restaurant.tsv
    2. Other stuffs
    3. Run Point 4.1 to 

### Usage of maps visualization


## 5. Advanced Search Engine
---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
