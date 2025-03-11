import json
import logging
import numpy as np
import time
from typing import List, Tuple
from openai import OpenAI
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

class PAS2:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-2024-08-06", embedding_model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model
        self.embedding_model = embedding_model
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def generate_paraphrased_questions(self, initial_question: str, n: int = 5) -> List[str]:
        paraphrased_questions = []
        existing_questions = [initial_question]
        for _ in range(n):
            new_question = self._generate_single_question(initial_question, existing_questions)
            paraphrased_questions.append(new_question)
            existing_questions.append(new_question)
            time.sleep(1)  # To avoid hitting rate limits
        return paraphrased_questions

    def _generate_single_question(self, initial_question: str, existing_questions: List[str]) -> str:
        try:
            existing_questions_str = "\n".join(existing_questions)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates paraphrases of the given question."
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Based on this question: '{initial_question}', "
                            f"generate a new question that is semantically similar but structurally different from the initial question. "
                            f"The new question should still lead to the same answer when asked about the context. "
                            f"The question should also be distinct from these existing questions:\n{existing_questions_str}\n\n"
                            "Provide only the new question, without any additional text."
                        )
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "single_question_generator",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"}
                            },
                            "required": ["question"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            json_response = response.choices[0].message.content
            parsed_response = json.loads(json_response)
            new_question = parsed_response["question"]
            self.logger.info(f"Generated paraphrased question: {new_question}")
            return new_question
        except Exception as e:
            self.logger.error(f"Error in generate_single_question: {e}")
            return "Failed to generate question"

    def get_llm_responses(self, questions: List[str]) -> List[str]:
        responses = []
        for question in questions:
            answer = self._get_llm_response(question)
            responses.append(answer)
            time.sleep(1)  # To avoid hitting rate limits
        return responses

    def _get_llm_response(self, question: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            answer = response.choices[0].message.content.strip()
            self.logger.info(f"LLM response for question: {question}\nResponse: {answer}")
            return answer
        except Exception as e:
            self.logger.error(f"Error in get_llm_response: {e}")
            return "Failed to get response"

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        try:
            texts = [text.replace("\n", " ") for text in texts]
            response = self.client.embeddings.create(input=texts, model=self.embedding_model)
            embeddings = [np.array(data.embedding) for data in response.data]
            return embeddings
        except Exception as e:
            self.logger.error(f"Error in get_embeddings: {e}")
            embedding_size = 1024  # Adjust if different
            return [np.zeros(embedding_size) for _ in texts]

    def calculate_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Symmetric matrix
        self.logger.info(f"Calculated similarity matrix:\n{similarity_matrix}")
        return similarity_matrix

    def is_semantically_trustworthy(self, similarity_matrix: np.ndarray, threshold: float = 0.9) -> bool:
        n = similarity_matrix.shape[0]
        total_pairs = 0
        pairs_above_threshold = 0
        
        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle to avoid counting pairs twice
                total_pairs += 1
                if similarity_matrix[i][j] >= threshold:
                    pairs_above_threshold += 1
        
        percentage_above_threshold = pairs_above_threshold / total_pairs if total_pairs > 0 else 0
        self.logger.info(f"Pairs above threshold: {pairs_above_threshold}/{total_pairs} = {percentage_above_threshold*100:.2f}%")
        
        return percentage_above_threshold >= 0.5  # Consider trustworthy if at least 50% are above threshold

    def compare_answers(self, answer1: str, answer2: str) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that determines whether two answers are coherent and contain the same information."
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Determine if the following two answers are coherent and contain the same information. Examine both answers thoroughly. Two answers are considered the same if they convey the same information, even if they are phrased differently or use different wording.\n\n"
                            f"Answer 1: '{answer1}'\n\nAnswer 2: '{answer2}'\n\n"
                            "Respond with a JSON object in the following format without any additional text:\n"
                            "{\"are_same\": true} or {\"are_same\": false}"
                        )
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_comparison",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "are_same": {"type": "boolean"}
                            },
                            "required": ["are_same"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            json_response = response.choices[0].message.content
            parsed_response = json.loads(json_response)
            are_same = parsed_response['are_same']
            self.logger.info(f"compare_answers result: {are_same}")
            return are_same
        except Exception as e:
            self.logger.error(f"Error in compare_answers: {e}")
            return False

    def answer_checker(self, answers: List[str]) -> np.ndarray:
        n = len(answers)
        match_matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                are_same = self.compare_answers(answers[i], answers[j])
                match_matrix[i][j] = are_same
                match_matrix[j][i] = are_same  # Symmetric matrix
                time.sleep(0.5)  # To avoid rate limits
            match_matrix[i][i] = True  # An answer is always the same as itself
        self.logger.info(f"Answer match matrix:\n{match_matrix}")
        return match_matrix

    def calculate_match_percentage(self, match_matrix: np.ndarray) -> float:
        n = match_matrix.shape[0]
        total_pairs = n * (n - 1) / 2
        matching_pairs = np.sum(np.triu(match_matrix, k=1))  # Sum of upper triangle excluding diagonal
        match_percentage = matching_pairs / total_pairs if total_pairs > 0 else 0
        self.logger.info(f"Matching pairs: {matching_pairs}, Total pairs: {total_pairs}, Match percentage: {match_percentage}")
        return match_percentage

    def save_matrix_plot(self, matrix: np.ndarray, labels: List[str], title: str, filename: str):
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
        plt.yticks(ticks=range(len(labels)), labels=labels)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        self.logger.info(f"Saved {title} plot to {filename}")

    def detect_hallucination(self, initial_question: str, n_paraphrases: int = 5, 
                             similarity_threshold: float = 0.9, 
                             match_percentage_threshold: float = 0.7) -> Tuple[bool, str, List[str], List[str]]:
        # Generate paraphrased questions
        paraphrased_questions = self.generate_paraphrased_questions(initial_question, n=n_paraphrases)
        all_questions = [initial_question] + paraphrased_questions

        # Get responses for all questions
        responses = self.get_llm_responses(all_questions)

        # Get embeddings for all responses
        embeddings = self.get_embeddings(responses)

        if len(embeddings) == 0:
            self.logger.error("Failed to get embeddings for responses.")
            return True, "Failed to process", all_questions, responses

        # Calculate semantic similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(embeddings)

        # Check semantic trustworthiness
        semantically_trustworthy = self.is_semantically_trustworthy(similarity_matrix, threshold=similarity_threshold)

        # Perform answer checking
        match_matrix = self.answer_checker(responses)
        match_percentage = self.calculate_match_percentage(match_matrix)
        answers_trustworthy = match_percentage >= match_percentage_threshold

        # Decide final trustworthiness
        trustworthy = semantically_trustworthy and answers_trustworthy
        hallucinated = not trustworthy

        # Generate labels for the matrices
        labels = ["original"] + [f"paraphrase{i}" for i in range(1, n_paraphrases + 1)]

        # Save similarity matrix plot
        sim_matrix_filename = f"similarity_matrix_{int(time.time())}.png"
        self.save_matrix_plot(similarity_matrix, labels, "Semantic Similarity Matrix", sim_matrix_filename)

        # Save answer match matrix plot
        match_matrix_filename = f"match_matrix_{int(time.time())}.png"
        self.save_matrix_plot(match_matrix.astype(float), labels, "Answer Match Matrix", match_matrix_filename)

        return hallucinated, responses[0], all_questions, responses