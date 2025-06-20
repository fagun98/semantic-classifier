"""
Example usage of SemanticModel: loading routes, using different encoders, training, and classifying queries.
This script demonstrates:
- How to load route definitions
- How to use different encoders (OpenAI, TF-IDF, Azure)
- How to train the SemanticModel
- How to classify queries
- How to evaluate model accuracy
"""

import os
import dotenv
from semantic_classifier import SemanticModel
from semantic_router.encoders import OpenAIEncoder, AzureOpenAIEncoder, TfidfEncoder

# Load environment variables (for API keys, etc.)
dotenv.load_dotenv('.env')

# Path to the routes file (update as needed)
ROUTES_PATH = "data/current_model/routes.json"

# 1. Load routes from file
# This loads the intent definitions (routes) from a JSON file.
routes = SemanticModel.load_routes(file_name=ROUTES_PATH)
print(f"Loaded {len(routes)} routes.")

# 2. Example: Using OpenAIEncoder
# This encoder uses OpenAI's API to generate embeddings for semantic search.
# Requires an OpenAI API key set in your environment variables.
openai_encoder = OpenAIEncoder(name="text-embedding-3-small")
# Initialize the SemanticModel with the OpenAI encoder and loaded routes
openai_model = SemanticModel(encoder=openai_encoder, router_file_path=ROUTES_PATH, train=True)
print("Training with OpenAIEncoder...")
openai_model.train_model()  # Train the model on the default data

# Classify a sample query using the trained model
query = "What is the attendance policy?"
result = openai_model.classify(query)
print(f"[OpenAIEncoder] Query: '{query}' => Intent: {result}")

# 3. Example: Using TfidfEncoder (offline, no API needed)
# This encoder uses TF-IDF, so it does not require any API keys or internet access.
tfidf_encoder = TfidfEncoder()
tfidf_encoder.fit(routes)  # Fit encoder to the loaded routes (required for TF-IDF)
# Initialize the SemanticModel with the TF-IDF encoder
# 'train=True' ensures the model is ready for training/classification
tfidf_model = SemanticModel(encoder=tfidf_encoder, router_file_path=ROUTES_PATH, train=True)
print("Training with TfidfEncoder...")
tfidf_model.train_model()

# Classify the same sample query using the TF-IDF model
result_tfidf = tfidf_model.classify(query)
print(f"[TfidfEncoder] Query: '{query}' => Intent: {result_tfidf}")

# 4. (Optional) Example: Using AzureOpenAIEncoder
# This encoder uses Azure's OpenAI service. Uncomment and set up your .env for Azure if needed.
# Make sure to provide the correct API key, deployment name, and endpoint in your .env file.
# azure_encoder = AzureOpenAIEncoder(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
# )
# azure_model = SemanticModel(encoder=azure_encoder, router_file_path=ROUTES_PATH, train=True)
# print("Training with AzureOpenAIEncoder...")
# azure_model.train_model()
# result_azure = azure_model.classify(query)
# print(f"[AzureOpenAIEncoder] Query: '{query}' => Intent: {result_azure}")