import heapq
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from unidecode import unidecode
import string
import pandas as pd
from collections import defaultdict
import numpy as np


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
    stops = set(stopwords.words('english'))  # stopwords
    tokens = [token for token in tokens if token not in stops]

    # Remove punctuation
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
    tokens = [token.replace("'", " ").replace("-", " ") for token in tokens]

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

    term = vocabulary_df['term'][term_id]  # get term from term_id
    n_term = len(inverted_index[term_id])  # number of documents that contain the term
    #print(n_term) # debugging line
    IDF = np.log10(n / n_term)  # calculate IDF of the term, inverse document frequency
    #print(f"IDF= {IDF}") # debugging line
    tf_idf_scores = []  # initialize list to store TF-IDF scores

    for doc_id in inverted_index[term_id]:
        #TF = preprocessed_docs[doc_id].count(term) / len(preprocessed_docs[doc_id]) # RELATIVE term frequency (tf)
        TF = preprocessed_docs[doc_id].count(term)  # raw term frequency (tf)
        #print(f"TF = {TF}")
        tf_idf_scores.append(TF * IDF)  # calculate TF-IDF score

    return tf_idf_scores


def top_k_restaurants(query, inverted_index, vocabulary_dict, doc_tf_idf_scores, df, n, k=5):
    '''
  Find the top k restaurants that match the given query using the TF-IDF scores
  Inputs:
  query: query string
  inverted_index: inverted index dictionary
  vocabulary_dict: dictionary containing the vocabulary of terms and their indeces
  doc_tf_idf_scores: dictionary storing the TF-IDF scores for each term in each document
  df: dataframe with restaurants data
  n: total number of documents
  k: number of restaurants to return
  Outputs:
  restaurants_df: dataframe with restaurants that match the query
  '''
    processed_query = preprocessing(query)  # processed query
    query_tokens = list(set(processed_query))  # unique query tokens
    #print(query_tokens) # debugging line
    # Find all docs to consider
    docs_to_consider = []  # initialize list to store documents to consider (non-zero intersection with the query tokens)

    for token in query_tokens:
        if vocabulary_dict[token]:  # check if the token is in the vocabulary
            token_id = vocabulary_dict[token]  # get the term_id of the token
            docs_to_consider.extend(
                inverted_index[token_id])  # add the documents that contain the token to the docs to consider

    docs_to_consider = list(set(docs_to_consider))  # remove duplicates

    # Calculate the TF-IDF score of the query
    query_tf_idf_scores = []  # initialize list to store the TF-IDF scores of the query
    for term in query_tokens:
        if term in vocabulary_dict.keys():
            term_id = vocabulary_dict[term]  # get the term_id of the term
            #print(inverted_index[term_id]) # debugging line
            n_term = len(inverted_index[term_id])  # number of documents that contain the term
            IDF = np.log10(n / n_term)  # calculate IDF of the term
            TF = processed_query.count(term)  # calculate TF of the term
            #print(f"TF = {TF}") # debugging line
            #print(f"IDF = {IDF}") # debuggin line
            query_tf_idf_scores.append((term_id, TF * IDF))  # calculate TF-IDF score

    query_tf_idf_scores.sort(key=lambda x: x[0])  # sort the query_tf_idf_scores in order of term_id

    query_norm = np.linalg.norm(
        np.array([score for _, score in query_tf_idf_scores]))  # calculate the norm of the query
    #print(f"query tf_idf_scores: {query_tf_idf_scores}") # debuggin line
    #print(f"query norm: {query_norm}") # debugging line
    # calculate document norms
    doc_norms = {doc_id: np.linalg.norm(
        np.array([doc_tf_idf_scores[doc_id][i][1] for i in range(len(doc_tf_idf_scores[doc_id]))])) for doc_id in
                 docs_to_consider}

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
        query_intersection = []  # initialize list to store (term, query_tf_idf) tuples in the intersection
        doc_intersection = []  # initialize list to store (term, doc_tf_idf) tuples in the intersection
        i, j = 0, 0  # initialize two pointers
        while i < len(query_tf_idf_scores) and j < len(doc_tf_idf_scores):
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
    cosine_similarity = defaultdict(float)  # initialize dictionary to store the cosine similarity results
    for doc_id in docs_to_consider:
        query_intersection, doc_intersection = query_doc_intersection(query_tf_idf_scores,
                                                                      doc_tf_idf_scores[doc_id])  # find the
        cosine_similarity[doc_id] = np.dot(np.array([score for _, score in query_intersection]),
                                           np.array([score for _, score in doc_intersection])) / (
                                                query_norm * doc_norms[doc_id])

    # Sort the cosine similarities in descending order
    sorted_cosine_similarity = sorted(cosine_similarity.items(), key=lambda x: x[1], reverse=True)  # list of tuples

    # Get the top k restaurants
    top_k_restaurants = sorted_cosine_similarity[:min(k, len(sorted_cosine_similarity))]

    top_k_restaurant_idx = [doc_id for doc_id, _ in top_k_restaurants]
    top_k_restaurant_scores = [score for _, score in top_k_restaurants]
    #print([score for _, score in top_k_restaurants])

    # build result dataframe
    restaurants_df = df.loc[top_k_restaurant_idx][['restaurantName', 'address', 'description', 'website']]
    restaurants_df['Similarity score'] = top_k_restaurant_scores

    return restaurants_df


def top_cosine_similarity(docs, query, docs_preprocessed=False, query_preprocessed=False):
    '''
  Function that preprocesses a list of documents and a query and returns
  the documents that match the query using cosine similarity
  Input:
  docs: pandas series of documents
  query: query string
  docs_preprocessed: True if the documents are already preprocessed and stored in a default dictionary
  query_preprocessed: True if the query is already preprocessed
  Output:
  sorted_cosine_similarity: list of tuples (doc_id, cosine_similarity_score) of the top k documents that match the query
  sorted by the cosine similarity obtained
  '''
    docs_copy = docs.copy()  # keep copy of original documents to calculate top k docs at the end
    n = len(docs)  # number of documents

    # Process documents if they're not already processed
    if not docs_preprocessed:
        preprocessed_docs = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            preprocessed_docs[doc_id] = preprocessing(doc)  # preprocess the description
    else:
        preprocessed_docs = docs  # store preprocessed docs here if already preprocessed

    # Process query if it's not already processed
    if not query_preprocessed:
        processed_query = preprocessing(query)  # preprocess query
    else:
        processed_query = query  # store preprocessed query here if already preprocessed

    # Find unique tokens
    doc_tokens = []  # initialize list to store all tokens
    for doc in preprocessed_docs.values():
        doc_tokens.extend(doc)
        doc_tokens = list(set(doc_tokens))  # remove duplicates

    # Define vocabulary dictionary and dataframe (mapping terms to their IDs)
    vocabulary_dict = {term: i for i, term in enumerate(doc_tokens)}  # dictionary of all vocabulary terms
    vocabulary_df = pd.DataFrame(
        {'term': vocabulary_dict.keys(), 'term_id': vocabulary_dict.values()})  # dataframe that maps terms to IDs

    # Compute inverted_index
    inverted_index = defaultdict(list)  # initialize inverted_index dictionary
    for doc_id, row in enumerate(docs):
        tokens = set(preprocessed_docs[doc_id])  # preprocessed description, eliminate duplicates
        for token in tokens:
            # Look up the term_id of the current term/token
            term_id = vocabulary_dict[token]
            # If the doc_id is not in the term_id's list in inverted_index, add it
            if doc_id not in inverted_index[term_id]:
                inverted_index[term_id].append(doc_id)

    # Compute updated_inverted_index
    n = len(preprocessed_docs)
    updated_inverted_index = defaultdict(
        list)  # initialize default dictionary to store the inverted_index values with TF-IDF scores
    inverted_index_copy = inverted_index.copy()  # Create a copy of the inverted_index to iterate over
    # fill updated_inverted_index
    for term_id, docs in inverted_index_copy.items():
        tf_idf_scores = tf_idf(int(term_id), inverted_index, preprocessed_docs, vocabulary_df, n)
        updated_inverted_index[term_id] = list(zip(docs, tf_idf_scores))

    # Retrieve TF-IDF scores of each document and store them in doc_tf_idf_scores
    doc_tf_idf_scores = defaultdict(list)  # initialize dictionary to store non-zero TF-IDF scores for each document
    for term_id, docs_scores in updated_inverted_index.items():
        for doc_id, tf_idf_score in docs_scores:
            if tf_idf_score != 0:
                doc_tf_idf_scores[doc_id].append((term_id, tf_idf_score))
        doc_tf_idf_scores[doc_id].sort(key=lambda x: x[0])  # sort the terms

    # Find all docs to consider
    docs_to_consider = []  # initialize list to store documents to consider (non-zero intersection with the query tokens)
    query_tokens = list(set(processed_query))  # unique query tokens
    # Get all docs that contain tokens from the query
    for token in query_tokens:
        if token in vocabulary_dict:  # check if the token is in the vocabulary
            token_id = vocabulary_dict[token]  # get the term_id of the token
            docs_to_consider.extend(
                inverted_index[token_id])  # add the documents that contain the token to the docs to consider
    docs_to_consider = list(set(docs_to_consider))  # remove duplicates

    # Calculate the TF-IDF score of the query
    query_tf_idf_scores = []  # initialize list to store the TF-IDF scores of the query
    for term in query_tokens:
        if term in vocabulary_dict.keys():
            term_id = vocabulary_dict[term]  # get the term_id of the term
            #print("II[termi_id]= ", inverted_index[term_id]) # debugging line
            n_term = len(inverted_index[term_id])  # number of documents that contain the term
            #print("n_term = ", n_term) # debugging lines
            IDF = np.log10(n / n_term)  # calculate IDF of the term
            #print(f"IDF= {IDF}") # debugging line
            TF = processed_query.count(term)  # calculate TF of the term
            query_tf_idf_scores.append((term_id, TF * IDF))  # calculate TF-IDF score
    query_tf_idf_scores.sort(key=lambda x: x[0])  # sort the query_tf_idf_scores in order of term_id

    # Query and document norms
    query_norm = np.linalg.norm(
        np.array([score for _, score in query_tf_idf_scores]))  # calculate the norm of the query
    doc_norms = {doc_id: np.linalg.norm(
        np.array([doc_tf_idf_scores[doc_id][i][1] for i in range(len(doc_tf_idf_scores[doc_id]))])) for doc_id in
                 docs_to_consider}  # calculate document norms

    # Function that returns two lists of tuples (term, query_tf_idf) and (term, doc_tf_idf) such that
    # the terms are in the intersection of the query terms and the doc's terms
    def query_doc_intersection(query_tf_idf_scores, doc_tf_idf_scores):
        '''
      Calculate the intersection of the query and the document with tf-idf scores
      Inputs:
      query_terms: list of sorted tuples of unique query terms and their tf-idf scores
      doc_terms: list of sorted tuples of unique document terms and their tf-idf scores
      Output:
      query_intersection: list of tuples (term, query_tf_idf)
      doc_intersection: list of tuples (term, doc_tf_idf)
      '''
        query_intersection = []  # initialize list to store (term, query_tf_idf) tuples in the intersection
        doc_intersection = []  # initialize list to store (term, doc_tf_idf) tuples in the intersection
        i, j = 0, 0  # initialize two pointers
        while i < len(query_tf_idf_scores) and j < len(doc_tf_idf_scores):
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
    cosine_similarity = defaultdict(float)  # initialize dictionary to store the cosine similarity results
    for doc_id in docs_to_consider:
        query_intersection, doc_intersection = query_doc_intersection(query_tf_idf_scores,
                                                                      doc_tf_idf_scores[doc_id])  # find the
        cosine_similarity[doc_id] = np.dot(np.array([score for _, score in query_intersection]),
                                           np.array([score for _, score in doc_intersection])) / (
                                                query_norm * doc_norms[doc_id])

    # Sort the cosine similarities in descending order
    sorted_cosine_similarity = sorted(cosine_similarity.items(), key=lambda x: x[1], reverse=True)  # list of tuples

    # Alternative outputs based on necessity
    #top_k_docs_df = pd.DataFrame([{'docName': docs_copy.loc[doc_id], 'similarity': score} for doc_id, score in top_k_docs_tuples])
    #top_k_docs_indices = [doc_id for doc_id, _ in top_k_docs_tuples]

    if not sorted_cosine_similarity:
        print("No documents found for the given query.")

    return sorted_cosine_similarity


def find_restaurants_updated(query_text, vocabulary_df, inverted_index, df):
    '''
    Find restaurants that match the given query using the inverted index
    Inputs:
    query: query string
    vocabulary_df: dataframe of vocabulary for query
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
        restaurants_df = df.loc[target_docs][:]  # Consider now all the columns

        # Return the DataFrame with the matching restaurants
        return restaurants_df

    except Exception as e:
        print(f"Error in finding restaurants: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


def find_top_custom_restaurants(query, vocabulary_df, inverted_index, df, k=5):
    '''
    Find top-k restaurants that match the given query using the inverted index and apply scoring.
    Inputs:
    query: dictionary with all the parameters inserted
    vocabulary_df: dataframe of vocabulary for query
    inverted_index: inverted index dictionary
    df: dataframe with restaurants data
    Outputs:
    top_k_df: dataframe with restaurants that match the query, orderd by the customScore created
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
    processed_query = preprocessing(query_text)
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
            # print(f'Debug: {TF} * {IDF}')
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
