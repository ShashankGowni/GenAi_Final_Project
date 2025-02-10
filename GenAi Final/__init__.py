



# import os
# import json
# import logging
# import azure.functions as func
# from openai import AzureOpenAI  # Azure OpenAI client
# from azure.search.documents import SearchClient
# from azure.search.documents.models import QueryType
# from azure.core.credentials import AzureKeyCredential

# # Initialize the Azure OpenAI client
# def initialize_openai_client():
#     # Ensure the environment variables are present and valid
#     endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION")
   
#     if not endpoint or not api_key or not api_version:
#         raise ValueError("One or more environment variables are missing or invalid.")
   
#     embedding_client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
#     return embedding_client
 
# # Initialize Azure Cognitive Search client
# def initialize_search_client():
#     # Ensure the environment variables are present and valid
#     endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
#     api_key = os.getenv("AZURE_SEARCH_KEY")
#     index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
   
#     if not endpoint or not api_key:
#         raise ValueError("One or more environment variables for Azure Search are missing or invalid.")
   
#     search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
#     return search_client
 
# # Generate embeddings for the query using Azure OpenAI's embedding model
# def generate_embedding(embedding_client, text):
#     emb = embedding_client.embeddings.create(model="text-embedding-3-small", input=text)
#     res = json.loads(emb.model_dump_json())  # Get the response in JSON format
#     return res["data"][0]["embedding"]  # Return the embedding
 
# # Perform vector similarity search in the Azure Cognitive Search index
# def perform_vector_search(search_client, query_vector, index_name, department=None):
#     search_client.index_name = index_name
   
#     # Creating the search parameters
#     search_parameters = {
#         "search": "*",  # Use wildcard for vector search
#         "vector": {
#             "value": query_vector,
#             "fields": "content_vector",  # The field to search
#             "k": 5  # Limit results to 5 to avoid large token counts
#         },
#         "top": 5  # Maximum number of documents to return
#     }

#     # Adding department filter if present
#     if department:
#         search_parameters["filter"] = f"Department eq '{department}'"

 
#     search_results = search_client.search(
#         search_text=search_parameters["search"],
#         vector=search_parameters["vector"],
#         filter=f"Department eq '{department}'",
#         top=search_parameters["top"]
#     )
#     return search_results
 
# # Generate the system message based on retrieved content
# def create_system_message(content_chunks):
#     total_length = 0
#     MAX_TOTAL_LENGTH = 1500  # Adjust this value as needed to fit within token limit
#     truncated_chunks = []

#     for chunk in content_chunks:
#         if total_length + len(chunk) > MAX_TOTAL_LENGTH:
#             truncated_chunks.append(chunk[:MAX_TOTAL_LENGTH - total_length])
#             break
#         truncated_chunks.append(chunk)
#         total_length += len(chunk)

#     system_message = "Answer the query based on the following content:"
#     content_message = "\n".join(truncated_chunks)
#     return f"{system_message}\n\nContent:\n{content_message}"

# # Estimate token count from message content
# def estimate_token_count(message_text):
#     words = " ".join(msg["content"] for msg in message_text).split()
#     tokens_per_word = 0.75  # Example ratio; adjust accordingly
#     return int(len(words) * tokens_per_word)

# # Query the knowledge base
# def main(req: func.HttpRequest) -> func.HttpResponse:
#     try:
#         # Check if the method is GET
#         if req.method != "GET":
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Method Not Allowed"}),
#                 status_code=405,
#                 mimetype="application/json"
#             )
 
#         # Get the query parameters
#         query = req.params.get("query")
#         department = req.params.get("Department")  # Optional 'Department' parameter to filter results
#         index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")  # Use the index from settings
#                 # Input validation
#         if not query or not index_name:
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Missing required query parameters: 'query' and 'index_name'"}),
#                 status_code=400,
#                 mimetype="application/json"
#             )
 
#         # Initialize OpenAI and Search clients
#         embedding_client = initialize_openai_client()
#         search_client = initialize_search_client()
 
#         # Step 1: Get vector for the query
#         query_vector = generate_embedding(embedding_client, query)
 
#         # Step 2: Perform vector similarity search in the given index
#         search_results = perform_vector_search(search_client, query_vector, index_name, department)
#         print(search_results)
        
#         if not search_results:
#             logging.warning("No search results returned.")
#             system_message = "No relevant content found. Please refine your query."
#             gpt_response = None
#             results_with_urls_and_deps = []
#         else:
#             MAX_CONTENT_LENGTH = 800
#             content_chunks = [result['content'][:MAX_CONTENT_LENGTH] for result in search_results]
        
        
        
#         # # Step 3: Check if content is available, and truncate content if needed
#         # MAX_CONTENT_LENGTH = 800  # Max characters per document to avoid exceeding token limit
#         # content_chunks = [result['content'][:MAX_CONTENT_LENGTH] for result in search_results]

#         # If no content found, use a default system message template
#         if content_chunks:
#             system_message = create_system_message(content_chunks)
#         else:
#             system_message = "No relevant content found. Please refine your query."
           
#         message_text = [{"role": "system", "content": system_message}, {"role": "user", "content": query}]
        
#         # Estimate token count to stay within limits
#         total_tokens = estimate_token_count(message_text)
        
#         if total_tokens > 16384:
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Query too lengthy. Please shorten your query or content."}),
#                 status_code=400,
#                 mimetype="application/json"
#             )

#         # Step 4: Send the query and system message to GPT model using Azure OpenAI client
#         response = embedding_client.chat.completions.create(
#             model="gpt-35-turbo-16k",  # GPT model with a larger context window
#             messages=message_text,
#             temperature=0.1  # Adjust temperature for more deterministic answers
#         )
 
#         # Step 5: Extract response from GPT
#         gpt_response = response.choices[0].message.content
        
        
#         results_with_urls_and_deps = []
        
#         for result in search_results:
            
#             doc_url = result.get("URL", "No URL available")
#             department = result.get("Department", "No Department")
#             content = result.get("content", "No Content")[:MAX_CONTENT_LENGTH]

#             document = {
#                     "user_query": query,
#                     "department": department,
#                     "document_url": doc_url,
#                     "content": content
#             }
#             results_with_urls_and_deps.append(document)

#         print("Final results with URLs and Departments:")
#         print(results_with_urls_and_deps)
#         # Return response
#         if gpt_response:
#             return func.HttpResponse(
#                 json.dumps({"response": gpt_response,"document_info": results_with_urls_and_deps,"error": None}),
#                 status_code=200,
#                 mimetype="application/json",
#                 headers={"Access-Control-Allow-Origin": "*"}
#             )
#         else:
#             return func.HttpResponse(
#                 json.dumps({"response": None,"document_info": results_with_urls_and_deps, "error": None}),
#                 status_code=200,
#                 mimetype="application/json",
#                 headers={"Access-Control-Allow-Origin": "*"}
#             )
 
#     except ValueError as ve:
#         logging.error(f"Configuration error: {str(ve)}")
#         return func.HttpResponse(
#             json.dumps({"response": None, "error": str(ve)}),
#             status_code=500,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )
#     except Exception as e:
#         logging.error(f"Error: {str(e)}")
#         return func.HttpResponse(
#             json.dumps({"response": None, "error": str(e)}),
#             status_code=500,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )

































 
 
# import os
# import json
# import logging
# import azure.functions as func
# from openai import AzureOpenAI  # Azure OpenAI client
# from azure.search.documents import SearchClient
# from azure.search.documents.models import QueryType
# from azure.core.credentials import AzureKeyCredential

# # Initialize the Azure OpenAI client
# def initialize_openai_client():
#     # Ensure the environment variables are present and valid
#     endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION")
   
#     if not endpoint or not api_key or not api_version:
#         raise ValueError("One or more environment variables are missing or invalid.")
   
#     embedding_client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
#     return embedding_client
 
# # Initialize Azure Cognitive Search client
# def initialize_search_client():
#     # Ensure the environment variables are present and valid
#     endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
#     api_key = os.getenv("AZURE_SEARCH_KEY")
#     index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
   
#     if not endpoint or not api_key:
#         raise ValueError("One or more environment variables for Azure Search are missing or invalid.")
   
#     search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
#     return search_client
 
# # Generate embeddings for the query using Azure OpenAI's embedding model
# def generate_embedding(embedding_client, text):
#     emb = embedding_client.embeddings.create(model="text-embedding-3-small", input=text)
#     res = json.loads(emb.model_dump_json())  # Get the response in JSON format
#     return res["data"][0]["embedding"]  # Return the embedding
 
# # Perform vector similarity search in the Azure Cognitive Search index
# def perform_vector_search(search_client, query_vector, index_name, department=None):
#     search_client.index_name = index_name
   
#     # Creating the search parameters
#     search_parameters = {
#         "search": "*",  # Use wildcard for vector search
#         "vector": {
#             "value": query_vector,
#             "fields": "content_vector",  # The field to search
#             "k": 5  # Limit results to 5 to avoid large token counts
#         },
#         "top": 5  # Maximum number of documents to return
#     }

#     # Adding department filter if present
#     if department:
#         search_parameters["filter"] = f"Department eq '{department}'"
 
#     search_results = search_client.search(search_parameters)
#     return search_results
 
# # Generate the system message based on retrieved content
# def create_system_message(content_chunks):
#     total_length = 0
#     MAX_TOTAL_LENGTH = 1500  # Adjust this value as needed to fit within token limit
#     truncated_chunks = []

#     for chunk in content_chunks:
#         if total_length + len(chunk) > MAX_TOTAL_LENGTH:
#             truncated_chunks.append(chunk[:MAX_TOTAL_LENGTH - total_length])
#             break
#         truncated_chunks.append(chunk)
#         total_length += len(chunk)

#     system_message = "Answer the query based on the following content:"
#     content_message = "\n".join(truncated_chunks)
#     return f"{system_message}\n\nContent:\n{content_message}"

# # Estimate token count from message content
# def estimate_token_count(message_text):
#     words = " ".join(msg["content"] for msg in message_text).split()
#     tokens_per_word = 0.75  # Example ratio; adjust accordingly
#     return int(len(words) * tokens_per_word)

# # Query the knowledge base
# def main(req: func.HttpRequest) -> func.HttpResponse:
#     try:
#         # Check if the method is GET
#         if req.method != "GET":
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Method Not Allowed"}),
#                 status_code=405,
#                 mimetype="application/json"
#             )
 
#         # Get the query parameters
#         query = req.params.get("query")
#         department = req.params.get("Department")  # Optional 'Department' parameter to filter results
#         index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")  # Use the index from settings
        
#         # Input validation
#         if not query or not index_name:
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Missing required query parameters: 'query' and 'index_name'"}),
#                 status_code=400,
#                 mimetype="application/json"
#             )
 
#         # Initialize OpenAI and Search clients
#         embedding_client = initialize_openai_client()
#         search_client = initialize_search_client()
 
#         # Step 1: Get vector for the query
#         query_vector = generate_embedding(embedding_client, query)
 
#         # Step 2: Perform vector similarity search in the given index
#         search_results = perform_vector_search(search_client, query_vector, index_name, department)
        
#         if not search_results:
#             logging.warning("No search results returned.")
#         else:
#             logging.debug(f"Search results: {search_results}")
        
#         logging.debug(f"Search results count: {len(list(search_results))}")
#         for result in search_results:
#             logging.debug(f"Search result: {result}")
 
#         # Step 3: Check if content is available, and truncate content if needed
#         MAX_CONTENT_LENGTH = 800  # Max characters per document to avoid exceeding token limit
#         content_chunks = [result['content'][:MAX_CONTENT_LENGTH] for result in search_results]

#         # If no content found, use a default system message template
#         if content_chunks:
#             system_message = create_system_message(content_chunks)
#         else:
#             system_message = "No relevant content found. Please refine your query."
           
#         message_text = [{"role": "system", "content": system_message}, {"role": "user", "content": query}]
        
#         # Estimate token count to stay within limits
#         total_tokens = estimate_token_count(message_text)
        
#         if total_tokens > 16384:
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Query too lengthy. Please shorten your query or content."}),
#                 status_code=400,
#                 mimetype="application/json"
#             )

#         # Step 4: Send the query and system message to GPT model using Azure OpenAI client
#         response = embedding_client.chat.completions.create(
#             model="gpt-35-turbo-16k",  # GPT model with a larger context window
#             messages=message_text,
#             temperature=0.1  # Adjust temperature for more deterministic answers
#         )
 
#         # Step 5: Extract response from GPT
#         gpt_response = response.choices[0].message.content
        
#         # Adding the doc URL
#         results_with_urls_and_deps = []
#         for result in search_results:
#             doc_url = result.get("URL", "No URL available")
#             department = result.get("Department", "No Department")
#             content = result.get("content", "No Content")[:MAX_CONTENT_LENGTH]  # Limit content length

#             logging.debug(f"Processing result: {result}")
#             print(f"Processing result: {result}")
#             print(f"Document URL: {doc_url}, Department: {department}, Content: {content}")
            
#             document = {
#                 "user_query": query,
#                 "department": department,
#                 "document_url": doc_url,
#                 "content": content
#             }
#             results_with_urls_and_deps.append(document)

#         print("Final results with URLs and Departments:")
#         print(results_with_urls_and_deps)
        
#         # Build the final response format
#         final_response = {
#             "response": gpt_response,
#             "department": department if department != "No Department" else None,
#             "URL": [doc["document_url"] for doc in results_with_urls_and_deps] if results_with_urls_and_deps else [],
#             "error": None
#         }
        
#         # Return response
#         return func.HttpResponse(
#             json.dumps(final_response),
#             status_code=200,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )
 
#     except ValueError as ve:
#         logging.error(f"Configuration error: {str(ve)}")
#         return func.HttpResponse(
#             json.dumps({"response": None, "department": None, "URL": None, "error": str(ve)}),
#             status_code=500,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )
#     except Exception as e:
#         logging.error(f"Error: {str(e)}")
#         return func.HttpResponse(
#             json.dumps({"response": None, "department": None, "URL": None, "error": str(e)}),
#             status_code=500,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )







#below is the testing and ongoing code

# import os
# import json
# import logging
# import azure.functions as func
# from openai import AzureOpenAI  # Azure OpenAI client
# from azure.search.documents import SearchClient
# from azure.search.documents.models import QueryType
# from azure.core.credentials import AzureKeyCredential

# # Initialize the Azure OpenAI client
# def initialize_openai_client():
#     # Ensure the environment variables are present and valid
#     endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION")
   
#     if not endpoint or not api_key or not api_version:
#         raise ValueError("One or more environment variables are missing or invalid.")
   
#     embedding_client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
#     return embedding_client
 
# # Initialize Azure Cognitive Search client
# def initialize_search_client():
#     # Ensure the environment variables are present and valid
#     endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
#     api_key = os.getenv("AZURE_SEARCH_KEY")
#     index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
   
#     if not endpoint or not api_key:
#         raise ValueError("One or more environment variables for Azure Search are missing or invalid.")
   
#     search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
#     return search_client
 
# # Generate embeddings for the query using Azure OpenAI's embedding model
# def generate_embedding(embedding_client, text):
#     emb = embedding_client.embeddings.create(model="text-embedding-3-small", input=text)
#     res = json.loads(emb.model_dump_json())  # Get the response in JSON format
#     return res["data"][0]["embedding"]  # Return the embedding
 
# # Perform vector similarity search in the Azure Cognitive Search index
# def perform_vector_search(search_client, query_vector, index_name, department=None):
#     search_client.index_name = index_name
   
#     # Creating the search parameters
#     search_parameters = {
#         "search": "*",  # Use wildcard for vector search
#         "vector": {
#             "value": query_vector,
#             "fields": "content_vector",  # The field to search
#             "k": 5  # Limit results to 5 to avoid large token counts
#         },
#         "top": 5  # Maximum number of documents to return
#     }
    
#         # Add category filter
#     search_filter = None
#     if department:
#         search_filter = f"department eq '{department}'"  # Make sure 'Category' is filterable in your index
#         logging.info(f"Search filter applied: {search_filter}")
 
#     search_results = search_client.search(
#         search_parameters,
#         filter=search_filter,
#         top=3
#         # Apply the filter to the search query
#     )

#     # # Adding department filter if present
#     # if department:
#     #     search_parameters["filter"] = f"department eq '{department}'"
 
#     # search_results = search_client.search(search_parameters)
#     return search_results
 
# # Generate the system message based on retrieved content
# def create_system_message(content_chunks):
#     total_length = 0
#     MAX_TOTAL_LENGTH = 1500  # Adjust this value as needed to fit within token limit
#     truncated_chunks = []

#     for chunk in content_chunks:
#         if total_length + len(chunk) > MAX_TOTAL_LENGTH:
#             truncated_chunks.append(chunk[:MAX_TOTAL_LENGTH - total_length])
#             break
#         truncated_chunks.append(chunk)
#         total_length += len(chunk)

#     system_message = "Answer the query based on the following content:"
#     content_message = "\n".join(truncated_chunks)
#     return f"{system_message}\n\nContent:\n{content_message}"

# # Estimate token count from message content
# def estimate_token_count(message_text):
#     words = " ".join(msg["content"] for msg in message_text).split()
#     tokens_per_word = 0.75  # Example ratio; adjust accordingly
#     return int(len(words) * tokens_per_word)




# def main(req: func.HttpRequest) -> func.HttpResponse:
#     try:
#         # Check if the method is GET
#         if req.method != "GET":
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Method Not Allowed"}),
#                 status_code=405,
#                 mimetype="application/json"
#             )

#         # Get the query parameters
#         query = req.params.get("query")
#         department = req.params.get("department")  # Optional 'Department' parameter to filter results
#         index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")  # Use the index from settings

#         # Input validation
#         if not query or not index_name:
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Missing required query parameters: 'query' and 'index_name'"}),
#                 status_code=400,
#                 mimetype="application/json"
#             )

#         # Initialize OpenAI and Search clients
#         embedding_client = initialize_openai_client()
#         search_client = initialize_search_client()

#         # Step 1: Get vector for the query
#         query_vector = generate_embedding(embedding_client, query)

#         # Step 2: Perform vector similarity search in the given index
#         search_results = perform_vector_search(search_client, query_vector, index_name, department)

#         if not search_results:
#             # If no relevant documents are found, respond with "No relevant content found."
#             return func.HttpResponse(
#                 json.dumps({"response": "No relevant content found.", "error": None}),
#                 status_code=200,
#                 mimetype="application/json",
#                 headers={"Access-Control-Allow-Origin": "*"}
#             )

#         # Step 3: Process the search results (fetch content directly)
#         results = []
#         MAX_CONTENT_LENGTH = 800  # Max characters per document to avoid exceeding token limit
        
        
#         content_chunks=[]
#         for result in search_results:
#             doc_url = result.get("URL", "No URL available")
#             department = result.get("department", "No Department")
#             content = result.get("content", "No Content")[:MAX_CONTENT_LENGTH]  # Limit content length
            
#             content_chunks.append(content)
#             # Create a dictionary for the document info
#             results.append({
#                 "document_url": doc_url,
#                 "department": department,
#                 "content": content
#             })
            


#         # Build the final response with the search results
#         final_response = {
#             "response": results,# List of results with content, URL, and department info
#             "error": None
#         }

#         # Return the results as a JSON response
#         return func.HttpResponse(
#             json.dumps(final_response),
#             status_code=200,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )

#     except ValueError as ve:
#         logging.error(f"Configuration error: {str(ve)}")
#         return func.HttpResponse(
#             json.dumps({"response": None, "department": None, "URL": None, "error": str(ve)}),
#             status_code=500,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )
#     except Exception as e:
#         logging.error(f"Error: {str(e)}")
#         return func.HttpResponse(
#             json.dumps({"response": None, "department": None, "URL": None, "error": str(e)}),
#             status_code=500,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )



# import os
# import json
# import logging
# import azure.functions as func
# from openai import AzureOpenAI  # Azure OpenAI client
# from azure.search.documents import SearchClient
# from azure.search.documents.models import QueryType
# from azure.core.credentials import AzureKeyCredential
# import numpy as np

# # Initialize the Azure OpenAI client
# def initialize_openai_client():
#     endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION")

#     if not endpoint or not api_key or not api_version:
#         raise ValueError("One or more environment variables are missing or invalid.")

#     embedding_client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
#     return embedding_client

# # Initialize Azure Cognitive Search client
# def initialize_search_client():
#     endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
#     api_key = os.getenv("AZURE_SEARCH_KEY")
#     index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

#     if not endpoint or not api_key:
#         raise ValueError("One or more environment variables for Azure Search are missing or invalid.")

#     search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
#     return search_client

# # Generate embeddings for the query using Azure OpenAI's embedding model
# def generate_embedding(embedding_client, text):
#     emb = embedding_client.embeddings.create(model="text-embedding-3-small", input=text)
#     res = json.loads(emb.model_dump_json())  # Get the response in JSON format
#     return np.array(res["data"][0]["embedding"])  # Return the embedding as a numpy array

# # Perform vector similarity search in the Azure Cognitive Search index
# def perform_vector_search(search_client, query_vector, index_name, department=None):
#     search_client.index_name = index_name

#     # Creating the search parameters
#     search_parameters = {
#         "search": "*",  # Use wildcard for vector search
#         "vector": {
#             "value": query_vector.tolist(),
#             "fields": "content_vector",  # The field to search
#             "k": 5  # Limit results to 5 to avoid large token counts
#         },
#         "top": 5  # Maximum number of documents to return
#     }

#     # Add category filter
#     search_filter = None
#     if department:
#         search_filter = f"department eq '{department}'"  # Make sure 'Category' is filterable in your index
#         logging.info(f"Search filter applied: {search_filter}")

#     search_results = search_client.search(
#         search_parameters,
#         filter=search_filter,
#         top=3  # Apply the filter to the search query
#     )

#     return search_results

# # Check if the content is relevant to the query
# def is_content_relevant(content, query):
#     # Simple keyword check for relevance
#     keywords = query.lower().split()
#     return any(keyword in content.lower() for keyword in keywords)

# # Generate the system message based on retrieved content
# def create_system_message(content_chunks):
#     total_length = 0
#     MAX_TOTAL_LENGTH = 1500  # Adjust this value as needed to fit within token limit
#     truncated_chunks = []

#     for chunk in content_chunks:
#         if total_length + len(chunk) > MAX_TOTAL_LENGTH:
#             truncated_chunks.append(chunk[:MAX_TOTAL_LENGTH - total_length])
#             break
#         truncated_chunks.append(chunk)
#         total_length += len(chunk)

#     system_message = "Answer the query based on the following content:"
#     content_message = "\n".join(truncated_chunks)
#     return f"{system_message}\n\nContent:\n{content_message}"

# # Estimate token count from message content
# def estimate_token_count(message_text):
#     words = " ".join(msg["content"] for msg in message_text).split()
#     tokens_per_word = 0.75  # Example ratio; adjust accordingly
#     return int(len(words) * tokens_per_word)

# def main(req: func.HttpRequest) -> func.HttpResponse:
#     try:
#         # Check if the method is GET
#         if req.method != "GET":
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Method Not Allowed"}),
#                 status_code=405,
#                 mimetype="application/json"
#             )

#         # Get the query parameters
#         query = req.params.get("query")
#         department = req.params.get("department")  # Optional 'Department' parameter to filter results
#         index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")  # Use the index from settings

#         # Input validation
#         if not query or not index_name:
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Missing required query parameters: 'query' and 'index_name'"}),
#                 status_code=400,
#                 mimetype="application/json"
#             )

#         # Initialize OpenAI and Search clients
#         embedding_client = initialize_openai_client()
#         search_client = initialize_search_client()

#         # Step 1: Get vector for the query
#         query_vector = generate_embedding(embedding_client, query)

#         # Step 2: Perform vector similarity search in the given index
#         search_results = perform_vector_search(search_client, query_vector, index_name, department)

#         if not search_results:
#             return func.HttpResponse(
#                 json.dumps({"response": "No relevant content found.", "error": None}),
#                 status_code=200,
#                 mimetype="application/json",
#                 headers={"Access-Control-Allow-Origin": "*"}
#             )

#         # Step 3: Process the search results (fetch content directly)
#         results = []
#         MAX_CONTENT_LENGTH = 800  # Max characters per document to avoid exceeding token limit

#         content_chunks = []
#         relevant_found = False
#         for result in search_results:
#             content = result.get("content", "No Content")[:MAX_CONTENT_LENGTH]  # Limit content length

#             if is_content_relevant(content, query):
#                 relevant_found = True
#                 doc_url = result.get("URL", "No URL available")
#                 department = result.get("department", "No Department")
#                 content_chunks.append(content)
                
#                 # Create a dictionary for the document info
#                 results.append({
#                     "document_url": doc_url,
#                     "department": department,
#                     "content": content
#                 })

#         # If no relevant content is found
#         if not relevant_found:
#             return func.HttpResponse(
#                 json.dumps({"response": "No relevant content found.", "error": None}),
#                 status_code=200,
#                 mimetype="application/json",
#                 headers={"Access-Control-Allow-Origin": "*"}
#             )

#         # Build the final response with the search results
#         final_response = {
#             "response": results,  # List of results with content, URL, and department info
#             "error": None
#         }

#         # Return the results as a JSON response
#         return func.HttpResponse(
#             json.dumps(final_response),
#             status_code=200,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )

#     except ValueError as ve:
#         logging.error(f"Configuration error: {str(ve)}")
#         return func.HttpResponse(
#             json.dumps({"response": None, "department": None, "URL": None, "error": str(ve)}),
#             status_code=500,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )
#     except Exception as e:
#         logging.error(f"Error: {str(e)}")
#         return func.HttpResponse(
#             json.dumps({"response": None, "department": None, "URL": None, "error": str(e)}),
#             status_code=500,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )





# #current working
# import os
# import json
# import logging
# import azure.functions as func
# from openai import AzureOpenAI  # Azure OpenAI client
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# import numpy as np

# # Initialize the Azure OpenAI client
# def initialize_openai_client():
#     endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION")

#     if not endpoint or not api_key or not api_version:
#         raise ValueError("One or more environment variables are missing or invalid.")

#     embedding_client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
#     return embedding_client

# # Initialize Azure Cognitive Search client
# def initialize_search_client():
#     endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
#     api_key = os.getenv("AZURE_SEARCH_KEY")
#     index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

#     if not endpoint or not api_key:
#         raise ValueError("One or more environment variables for Azure Search are missing or invalid.")

#     search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
#     return search_client

# # Generate embeddings for the query using Azure OpenAI's embedding model
# def generate_embedding(embedding_client, text):
#     emb = embedding_client.embeddings.create(model="text-embedding-3-small", input=text)
#     res = json.loads(emb.model_dump_json())  # Get the response in JSON format
#     return np.array(res["data"][0]["embedding"])  # Return the embedding as a numpy array

# # Perform vector similarity search in the Azure Cognitive Search index
# def perform_vector_search(search_client, query_vector, index_name, department=None):
#     search_client.index_name = index_name

#     # Creating the search parameters
#     search_parameters = {
#         "search": "*",  # Use wildcard for vector search
#         "vector": {
#             "value": query_vector.tolist(),
#             "fields": "content_vector",  # The field to search
#             "k": 5  # Limit results to 5 to avoid large token counts
#         },
#         "top": 5  # Maximum number of documents to return
#     }

#     # Add category filter
#     search_filter = None
#     if department:
#         search_filter = f"department eq '{department}'"  # Make sure 'Category' is filterable in your index
#         logging.info(f"Search filter applied: {search_filter}")

#     search_results = search_client.search(
#         search_parameters,
#         filter=search_filter,
#         top=3  # Apply the filter to the search query
#     )

#     return search_results

# # Check if the content is relevant to the query
# def is_content_relevant(content, query):
#     # Simple keyword check for relevance
#     keywords = query.lower().split()
#     return any(keyword in content.lower() for keyword in keywords)

# # Generate the system message based on retrieved content
# def create_system_message(content_chunks):
#     total_length = 0
#     MAX_TOTAL_LENGTH = 1500  # Adjust this value as needed to fit within token limit
#     truncated_chunks = []

#     for chunk in content_chunks:
#         if total_length + len(chunk) > MAX_TOTAL_LENGTH:
#             truncated_chunks.append(chunk[:MAX_TOTAL_LENGTH - total_length])
#             break
#         truncated_chunks.append(chunk)
#         total_length += len(chunk)

#     system_message = "Answer the query based on the following content:"
#     content_message = "\n".join(truncated_chunks)
#     return f"{system_message}\n\nContent:\n{content_message}"

# # Estimate token count from message content
# def estimate_token_count(message_text):
#     words = " ".join(msg["content"] for msg in message_text).split()
#     tokens_per_word = 0.75  # Example ratio; adjust accordingly
#     return int(len(words) * tokens_per_word)

# # # Generate GPT response based on the system message
# # def generate_gpt_response(embedding_client, content_chunks, query):
# #     # Create system message
# #     system_message = create_system_message(content_chunks)

# #     # Now use the system message and pass it to GPT
# #     prompt = f"{system_message}\n\nUser Query: {query}\n\nResponse:"
# #     message_text = [{"role": "system", "content": system_message}, {"role": "user", "content": query}]
        
# #     # Call the GPT model to generate a response based on the prompt
# #     try:
# #         response = embedding_client.chat.completions.create(
# #             model="gpt-35-turbo-16k",  # Choose the model you want
# #             messages=message_text,
# #             max_tokens=200,  # Set a reasonable token limit based on your needs
# #             temperature=0.5  # Control response creativity (0.0 to 1.0)
# #         )
# #         # gpt_response = response.choices.text.strip()
# #         return response
# #     except Exception as e:
# #         logging.error(f"Error generating GPT response: {str(e)}")
# #         return "Error generating response from GPT."
# def generate_gpt_response(embedding_client, content_chunks, query):
#     # Create system message
#     system_message = create_system_message(content_chunks)

#     # Now use the system message and pass it to GPT
#     prompt = f"{system_message}\n\nUser Query: {query}\n\nResponse:"
#     message_text = [{"role": "system", "content": system_message}, {"role": "user", "content": query}]
        
#     # Call the GPT model to generate a response based on the prompt
#     try:
#         response = embedding_client.chat.completions.create(
#             model="gpt-35-turbo-16k",  # Choose the model you want
#             messages=message_text,
#             max_tokens=200,  # Set a reasonable token limit based on your needs
#             temperature=0.5  # Control response creativity (0.0 to 1.0)
#         )

#         # Correctly extract the GPT response from the choices
#         gpt_response = response.choices[0].message.content  # This part needs to be changed

#         return gpt_response
#     except Exception as e:
#         logging.error(f"Error generating GPT response: {str(e)}")
#         return "Error generating response from GPT."

# def main(req: func.HttpRequest) -> func.HttpResponse:
#     try:
#         # Check if the method is GET
#         if req.method != "GET":
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Method Not Allowed"}),
#                 status_code=405,
#                 mimetype="application/json"
#             )

#         # Get the query parameters
#         query = req.params.get("query")
#         department = req.params.get("department")  # Optional 'Department' parameter to filter results
#         index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")  # Use the index from settings

#         # Input validation
#         if not query or not index_name:
#             return func.HttpResponse(
#                 json.dumps({"response": None, "error": "Missing required query parameters: 'query' and 'index_name'"}),
#                 status_code=400,
#                 mimetype="application/json"
#             )

#         # Initialize OpenAI and Search clients
#         embedding_client = initialize_openai_client()
#         search_client = initialize_search_client()

#         # Step 1: Get vector for the query
#         query_vector = generate_embedding(embedding_client, query)

#         # Step 2: Perform vector similarity search in the given index
#         search_results = perform_vector_search(search_client, query_vector, index_name, department)

#         if not search_results:
#             return func.HttpResponse(
#                 json.dumps({"response": "No relevant content found.", "error": None}),
#                 status_code=200,
#                 mimetype="application/json",
#                 headers={"Access-Control-Allow-Origin": "*"}
#             )

#         # Step 3: Process the search results (fetch content directly)
#         results = []
#         MAX_CONTENT_LENGTH = 800  # Max characters per document to avoid exceeding token limit

#         content_chunks = []
#         relevant_found = False
#         for result in search_results:
#             content = result.get("content", "No Content")[:MAX_CONTENT_LENGTH]  # Limit content length

#             if is_content_relevant(content, query):
#                 relevant_found = True
#                 doc_url = result.get("URL", "No URL available")
#                 department = result.get("department", "No Department")
#                 content_chunks.append(content)
                
#                 # Create a dictionary for the document info
#                 results.append({
#                     "document_url": doc_url,
#                     "department": department,
#                     "content": content
#                 })

#         # If no relevant content is found
#         if not relevant_found:
#             return func.HttpResponse(
#                 json.dumps({"response": "No relevant content found.", "error": None}),
#                 status_code=200,
#                 mimetype="application/json",
#                 headers={"Access-Control-Allow-Origin": "*"}
#             )

            
        
        

#         # Step 4: Generate GPT response based on the content chunks
#         gpt_response = generate_gpt_response(embedding_client, content_chunks, query)

#         # Step 5: Append document links below GPT response
#         document_links = "\n\nHere are some related documents for your reference:\n"
#         for result in results:
#             document_links += f" - {result['document_url']}\n"

#         # Final response including GPT-generated answer and document links
#         final_response = {
#             "response": gpt_response,
#             'documents_links':document_links,# GPT response + document links
#             "error": None
#         }

#         # Return the results as a JSON response
#         return func.HttpResponse(
#             json.dumps(final_response),
#             status_code=200,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )

#     except ValueError as ve:
#         logging.error(f"Configuration error: {str(ve)}")
#         return func.HttpResponse(
#             json.dumps({"response": None, "error": str(ve)}),
#             status_code=500,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )
#     except Exception as e:
#         logging.error(f"Error: {str(e)}")
#         return func.HttpResponse(
#             json.dumps({"response": None, "error": str(e)}),
#             status_code=500,
#             mimetype="application/json",
#             headers={"Access-Control-Allow-Origin": "*"}
#         )


import os
import json
import logging
import azure.functions as func
from openai import AzureOpenAI  # Azure OpenAI client
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import numpy as np

# Initialize the Azure OpenAI client
def initialize_openai_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not endpoint or not api_key or not api_version:
        raise ValueError("One or more environment variables are missing or invalid.")

    embedding_client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    return embedding_client

# Initialize Azure Cognitive Search client
def initialize_search_client():
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

    if not endpoint or not api_key:
        raise ValueError("One or more environment variables for Azure Search are missing or invalid.")

    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
    return search_client

# Generate embeddings for the query using Azure OpenAI's embedding model
def generate_embedding(embedding_client, text):
    emb = embedding_client.embeddings.create(model="text-embedding-3-small", input=text)
    res = json.loads(emb.model_dump_json())
    logging.info(np.array(res["data"][0]["embedding"]))# Get the response in JSON format
    return np.array(res["data"][0]["embedding"])  # Return the embedding as a numpy array

# Perform vector similarity search in the Azure Cognitive Search index
def perform_vector_search(search_client, query_vector, index_name, department=None):
    search_client.index_name = index_name

    # Creating the search parameters
    search_parameters = {
        "search": "*",  # Use wildcard for vector search
        "vector": {
            "value": query_vector.tolist(),
            "fields": "vector",  # The field to search
            "k": 5  # Limit results to 5 to avoid large token counts
        
        },
        "top": 5  # Maximum number of documents to return
    }

    # Add category filter
    search_filter = None
    if department:
        search_filter = f"department eq '{department}'"  # Make sure 'Category' is filterable in your index
        logging.info(f"Search filter applied: {search_filter}")

    search_results = search_client.search(
        search_parameters,
        filter=search_filter,
        top=10  # Apply the filter to the search query
    )

    return search_results

# Check if the content is relevant to the query
def is_content_relevant(content, query, embedding_client):
    # Generate embeddings for both content and query
    content_embedding = generate_embedding(embedding_client, content)
    query_embedding = generate_embedding(embedding_client, query)
    
    # Calculate cosine similarity between the content and the query
    similarity = np.dot(content_embedding, query_embedding) / (np.linalg.norm(content_embedding) * np.linalg.norm(query_embedding))
    
    # Define a strict threshold for relevance (e.g., 0.8 or higher for more relevant content)
    similarity_threshold = 0.2
    logging.info(f"Cosine similarity: {similarity}")
    return similarity >= similarity_threshold

# Generate the system message based on retrieved content
def create_system_message(content_chunks):
    total_length = 0
    MAX_TOTAL_LENGTH = 1500  # Adjust this value as needed to fit within token limit
    truncated_chunks = []

    for chunk in content_chunks:
        if total_length + len(chunk) > MAX_TOTAL_LENGTH:
            truncated_chunks.append(chunk[:MAX_TOTAL_LENGTH - total_length])
            break
        truncated_chunks.append(chunk)
        total_length += len(chunk)

    system_message = "Answer user query based on the below contents and if the user query is not related to the below content response no relevant content found:"
    content_message = "\n".join(truncated_chunks)
    return f"{system_message}\n\nContent:\n{content_message}"

# Generate GPT response based on the system message
def generate_gpt_response(embedding_client, content_chunks, query):
    # Create system message
    system_message = create_system_message(content_chunks)

    # Prepare the prompt for GPT
    message_text = [{"role": "system", "content": system_message}, {"role": "user", "content": query}]
        
    # Call the GPT model to generate a response based on the prompt
    try:
        response = embedding_client.chat.completions.create(
            model="gpt-35-turbo-16k",  # Choose the model you want
            messages=message_text,
            max_tokens=200,  # Set a reasonable token limit based on your needs
            temperature=0.5  # Control response creativity (0.0 to 1.0)
        )

        # Correctly extract the GPT response from the choices
        gpt_response = response.choices[0].message.content  # This part needs to be changed

        return gpt_response
    except Exception as e:
        logging.error(f"Error generating GPT response: {str(e)}")
        return "Error generating response from GPT."

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Check if the method is GET
        if req.method != "GET":
            return func.HttpResponse(
                json.dumps({"response": None, "error": "Method Not Allowed"}),
                status_code=405,
                mimetype="application/json"
            )

        # Get the query parameters
        query = req.params.get("query")
        department = req.params.get("department")  # Optional 'Department' parameter to filter results
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")  # Use the index from settings

        # Input validation
        if not query or not index_name:
            return func.HttpResponse(
                json.dumps({"response": None, "error": "Missing required query parameters: 'query' and 'index_name'"}),
                status_code=400,
                mimetype="application/json"
            )

        # Initialize OpenAI and Search clients
        embedding_client = initialize_openai_client()
        search_client = initialize_search_client()

        # Step 1: Get vector for the query
        query_vector = generate_embedding(embedding_client, query)

        # Step 2: Perform vector similarity search in the given index
        search_results = perform_vector_search(search_client, query_vector, index_name, department)

        if not search_results:
            return func.HttpResponse(
                json.dumps({"response": "No relevant content found.", "error": None}),
                status_code=200,
                mimetype="application/json",
                headers={"Access-Control-Allow-Origin": "*"}
            )

        # Step 3: Process the search results (fetch content directly)
        results = []
        MAX_CONTENT_LENGTH = 800  # Max characters per document to avoid exceeding token limit

        content_chunks = []
        relevant_found = False
        for result in search_results:
            content = result.get("content", "No Content")[:MAX_CONTENT_LENGTH]  # Limit content length

            if is_content_relevant(content, query, embedding_client):
                relevant_found = True
                doc_url = result.get("URL", "No URL available")
                department = result.get("department", "No Department")
                content_chunks.append(content)
                
                # Create a dictionary for the document info
                results.append({
                    "document_url": doc_url,
                    "department": department,
                    "content": content
                })

        # If no relevant content is found
        if not relevant_found:
            return func.HttpResponse(
                json.dumps({"response": "No relevant content found.", "error": None}),
                status_code=200,
                mimetype="application/json",
                headers={"Access-Control-Allow-Origin": "*"}
            )

        # Step 4: Generate GPT response based on the content chunks
        gpt_response = generate_gpt_response(embedding_client, content_chunks, query)

        # Step 5: Append document links below GPT response
        document_links = "\n\nHere are some related documents for your reference:\n"
        for result in results:
            department_text = f" (Department: {result['department']})" if result['department'] else " (Department: N/A)"
            document_links += f" - {result['document_url']}{department_text}\n"

        # Final response including GPT-generated answer and document links
        final_response = {
            "response": gpt_response,
            'documents_links': document_links,  # GPT response + document links
            "error": None
        }

        # Return the results as a JSON response
        return func.HttpResponse(
            json.dumps(final_response),
            status_code=200,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except ValueError as ve:
        logging.error(f"Configuration error: {str(ve)}")
        return func.HttpResponse(
            json.dumps({"response": None, "error": str(ve)}),
            status_code=500,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"response": None, "error": str(e)}),
            status_code=500,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )
