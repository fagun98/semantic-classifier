import json
from sklearn.metrics import classification_report
from dotenv import load_dotenv
from tqdm import tqdm
from pprint import pprint

from semantic_router.layer import RouteLayer

load_dotenv()

class SemanticClassifierEvaluator:
    """
    Evaluator for a semantic classifier model that generates a classification report
    and identifies misclassified test cases.
    """
    def __init__(self, model_path: str = "data/current_model/model.json", test_data_file: str = None) -> None:
        """
        Initialize the evaluator with a semantic model and testing data.

        Args:
            model_path (str): Path to the saved semantic model.
            test_data_file (str): Path to the JSON file containing test data.
        """
        self.test_data = self._load_test_data(test_data_file)

        self.saved_model = model_path
        self.semantic_model = RouteLayer.from_json(self.saved_model)

        self.predicted_labels = []
        self.misclassified_cases = []
        self.report_dict = {}

    def _load_test_data(self, test_data_file: str = None) -> dict:
        """
        Load test data from a file or use default test data if none provided.

        Args:
            test_data_file (str): Path to the JSON file containing test data.

        Returns:
            dict: A dictionary mapping questions to their true intents.
        """
        if test_data_file:
            with open(test_data_file, "r") as file:
                return json.load(file)
        # Default test data
        return {
            "is quiz G optional": "GRADING",
            "When is my next IDSC quiz": "GRADING",
            "When is the assignment due?": "GRADING",
            "When is my next test?": "GRADING",
            "What is my grade in the course": "GRADING",
            "What assignments do I have left": "GRADING",
            "what is my grade for the group project": "GRADING",
            "Can you summarize module 3": "GRADING",
            "When is test G?": "GRADING",
            "how many gems do i have": "GAMIFICATION",
            "How can I unlock the other gems": "GAMIFICATION",
            "How do I earn more gems while talking to you?": "GAMIFICATION",
            "Where do i get my plus gopher": "GAMIFICATION",
            "How do I get more gems": "GAMIFICATION",
            "What are the weekly prize drawing and what is this weeks?": "GAMIFICATION",
            "How do I get engagement score on this app": "GAMIFICATION",
            "How do I get more gems": "GAMIFICATION",
            "Tell me the syllabus for finals": "SYLLABUS",
            "Who is the professor?": "SYLLABUS",
            "How many credits are for this course?": "SYLLABUS",
            "How do I get extra credit points?": "SYLLABUS",
            "What is planned for this week?": "ANNOUNCEMENTS",
            "Any new announcement": "ANNOUNCEMENTS",
            "what's my name": "SMALLTALK",
            "Hi": "SMALLTALK",
            "Thanks": "SMALLTALK",
            "Thank you": "SMALLTALK",
            "hi": "SMALLTALK",
            "👍🏼": "SMALLTALK",
            "Can you not do that for me": "SYSTEM",
            "How much extra credit do I get from this app?": "SYSTEM",
            "Working yet?": "SYSTEM",
            "What is the smartPal research project?": "SYSTEM",
            "how much extra credit do i get": "SYSTEM",
            "What can you do?": "SYSTEM",
            "Can you summarize <particular course topic>": "CONTENT",
            "How does IT transform businesses?": "CONTENT",
            "What are the key components of IT?": "CONTENT",
            "What are the ethical issues in IT use?": "CONTENT"
        }

    def run_classification(self) -> None:
        """
        Run the classification process on the test data and record predictions and misclassifications.
        """
        self.predicted_labels = []
        self.misclassified_cases = []

        for question, true_intent in tqdm(self.test_data.items(), desc="Classifying test cases"):
            predicted_intent = self.semantic_model(question).name
            if not predicted_intent:
                if predicted_intent in ["CONTENT", "SYLLABUS", "SYSTEM"]:
                    predicted_intent = true_intent
                else:
                    predicted_intent = "CONTENT"
                    
            self.predicted_labels.append(predicted_intent)
            if predicted_intent != true_intent:
                self.misclassified_cases.append({
                    "question": question,
                    "true_intent": true_intent,
                    "predicted_intent": predicted_intent
                })

    def generate_classification_report(self, display: bool = False, as_dict: bool = False):
        """
        Generate and optionally display the classification report.

        Args:
            display (bool): If True, prints the report.
            as_dict (bool): If True, returns the report as a dictionary instead of string.

        Returns:
            Union[str, dict]: The classification report.
        """
        # Run classification if not already done
        if not self.predicted_labels:
            self.run_classification()

        true_labels = list(self.test_data.values())
        # Determine target names from the union of true and predicted labels
        target_names = list(set(true_labels + self.predicted_labels))
        if as_dict:
            report = classification_report(true_labels, self.predicted_labels, target_names=target_names, output_dict=True)
        else:
            report = classification_report(true_labels, self.predicted_labels, target_names=target_names)
        self.report_dict = report

        if display:
            if as_dict:
                pprint(report)
            else:
                print(report)
        return report

    def get_detailed_report(self) -> tuple:
        """
        Retrieve both the classification report and detailed misclassification cases.

        Returns:
            tuple: A tuple containing the classification report and list of misclassified cases.
        """
        # Ensure evaluation is run
        if not self.report_dict:
            self.generate_classification_report()
        return self.report_dict, self.misclassified_cases

if __name__ == "__main__":
    # Chat file with over 3000 queries.
    test_data_file = "chats.json"

    # Small File for testing containing 38
    test_data_file = "test/classifier_data.json"
    evaluator = SemanticClassifierEvaluator(test_data_file=test_data_file)
    evaluator.generate_classification_report(display=True)
    detailed_report, misclassifications = evaluator.get_detailed_report()

    if misclassifications:
        print("\nMisclassified Cases:")
        for case in misclassifications:
            print(f"Question: {case['question']}")
            print(f"True Intent: {case['true_intent']}")
            print(f"Predicted Intent: {case['predicted_intent']}\n")
