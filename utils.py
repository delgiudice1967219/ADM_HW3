import os
import glob
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
                    # print(f"Processing: {filename}")
                    file_path = os.path.join(folder_path, filename)
                    restaurant_info = parser.extract_restaurant_info(file_path)
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
