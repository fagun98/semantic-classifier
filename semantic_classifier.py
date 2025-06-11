# Import necessary modules from the semantic_router package
from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer
import json
import dotenv 

# Load environment variables from the .env file
dotenv.load_dotenv('.env')

class SemanticModel:
    """
    A class to represent a semantic model that handles route loading, training, 
    and classification tasks using a given encoder and route layer.
    """

    def __init__(self, encoder=None, router_file_path:str = None, saved_model:str = None, train:bool = False) -> None:
        """
            Initializes the SemanticModel with routes and an encoder.
            If no encoder is provided, it defaults to OpenAIEncoder.
            
            `saved_model`:str - Path to saved model (a json file)
        """

        self.saved_model = 'data/current_model/model.json'
        self.router_file_path = 'data/current_model/routes.json'
        
        if saved_model:
            self.saved_model = saved_model

        if router_file_path:
            self.router_file_path = router_file_path

        if train:
            # Load route data from JSON and initialize routes
            self.routes_data = self.load_routes(file_name=self.router_file_path)
            self.routes = [Route(**route) for route in self.routes_data]
            
            # Initialize encoder (use OpenAIEncoder if none provided)
            self.encoder = encoder if encoder else OpenAIEncoder()
            
            # Create a RouteLayer model using the encoder and loaded routes
            self.model = RouteLayer(encoder=self.encoder, routes=self.routes)
            
            #train model
            # self.train_model()
        
        else:
            self.model = RouteLayer.from_json(self.saved_model)

    def save_model(self):
        """
            save the model to json file.
        """
        self.model.to_json(self.saved_model)

    def load_routes(self, file_name: str = None):
        """
        Loads route configurations from a specified JSON file.
        
        Args:
            file_name (str): The name of the JSON file containing route data.
        
        Returns:
            list: A list of dictionaries representing route configurations.
        """
        with open(file_name, 'r', encoding='utf-8') as file:
            intents = json.load(file)
        return intents
    
    def load_test_data(self, file_name: str = 'classifier/intents.json'):
        """
        Loads test data from a specified JSON file for model evaluation.
        
        Args:
            file_name (str): The name of the JSON file containing test data.
        
        Returns:
            tuple: A tuple containing two lists, data_x (input data) and label_y (expected labels).
        """
        with open(file_name, 'r', encoding='utf-8') as file:
            test_data_file = json.load(file)
    
        test_data = test_data_file['rows']

        # Extract messages (input data) and true intents (labels) from test data
        data_x = []
        label_y = []
        for rows in test_data:
            data_x.append(rows['message'])
            label_y.append(rows['true_intent'])
        
        return data_x, label_y
    
    def get_accuracy(self):
        """
        Evaluates the model's accuracy based on the test data.
        
        Returns:
            str: A formatted string displaying the model's accuracy as a percentage.
        """
        # Load test data
        data_x, label_y = self.load_test_data()
        
        # Evaluate model accuracy
        accuracy = self.model.evaluate(X=data_x, y=label_y)
        
        return f"Accuracy: {accuracy*100:.2f}%"

    def get_threshold(self):
        """
        Retrieves and prints the model's classification thresholds.
        
        Returns:
            None
        """
        # Fetch and print the thresholds from the model
        self.model.get_thresholds()

    def train_model(self, train_data_file: str = None):
        """
        Trains the model using the provided training data file, or defaults to the test data.
        
        Args:
            train_data_file (str): Optional; The name of the JSON file containing training data.
        
        Returns:
            dict: Updated classification thresholds after training.
        """
        data_x, label_y = None, None

        # Load training data from the specified file, or default to test data
        if train_data_file:
            data_x, label_y = self.load_test_data(file_name=train_data_file)
        else:
            data_x, label_y = self.load_test_data()

        # Train the model with the provided data
        self.model.fit(X=data_x, y=label_y)
        
        # Retrieve and return updated thresholds after training
        updated_thresholds = self.model.get_thresholds()
        return updated_thresholds
        
    def get_semantic_model(self):
        """
        Returns the underlying semantic model for external use.
        
        Returns:
            RouteLayer: The RouteLayer model instance.
        """
        return self.model

    def classify(self, query: str):
        """
        Classifies a given query using the trained semantic model.
        
        Args:
            query (str): The input query string to classify.
        
        Returns:
            str: The name of the classified intent.
        """
        return self.model(query).name
    
    def extended_classify(self, query: str, previous_question:str = None, score_threshold:float = 0.6):
        """
        Classifies a given query using the trained semantic model.
        
        Args:
            query (str): The input query string to classify.
        
        Returns:
            dict: The classification result from the model.
        """
        classified_result = self.model.retrieve_multiple_routes(query)

        if not classified_result and not previous_question: 
            return "LLM"
        
        classified_result = sorted(
                                    classified_result,
                                    key=lambda rc: rc.similarity_score,
                                    reverse=True
                                )

        classified_intent = classified_result[0].name
        classified_score = classified_result[0].similarity_score

        if previous_question is None:
            if classified_score < score_threshold:
                return "LLM"
            else:
                return classified_intent
        
        combined_query = previous_question + " " + query
        combined_classified_result = self.model.retrieve_multiple_routes(combined_query)
        combined_classified_result = sorted(
                                    combined_classified_result,
                                    key=lambda rc: rc.similarity_score,
                                    reverse=True
                                )
        combined_classified_intent = combined_classified_result[0].name
        combined_classified_score = combined_classified_result[0].similarity_score

        if classified_score < score_threshold and combined_classified_score < score_threshold:
            return "LLM"
        elif classified_score < score_threshold:
            return combined_classified_intent
        else:
            if combined_classified_score > classified_score:
                return combined_classified_intent
            else:
                return classified_intent

if __name__ == "__main__":
    """
    Main execution block: Initialize the SemanticModel, display model thresholds, 
    train the model, and evaluate it on predefined questions.
    """
    
    # Initialize the semantic model
    rl = SemanticModel()

    # Display the model's classification thresholds
    
    # print("\n==========={threshold}=========\n")
    # print(rl.get_threshold())

    # Train the model and display the updated thresholds
    
    # print("\n==========={train_model}=========\n")
    # print(rl.train_model())

    # Display the model's accuracy on the test data
    
    # print("\n==========={accuracy}=========\n")
    # print(rl.get_accuracy())

    # List of sample questions for classification
    questions = [
        "What chapter in the textbook are we focusing on this week?",
        "Pal, what topics are covered on Test A?",
        "Hi",
        "what letter grade is that",
        "Who is my professor?",
        "What does canvas view pages mean",
        "What time is my class?",
        "How do I improve that",
        "What is the attendance policy",
        "When’s next class",
        "After the final, what is my final grade roughly?",
        "What did I get on the last exam",
        "How many exams are there?",
        "What is the weight of tests?",    
    ]

    # Classify each question using the model and print the results
    for question in questions:
        print(f"\n {question} : {rl.classify(question)}\n")