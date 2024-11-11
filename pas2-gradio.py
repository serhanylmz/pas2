import gradio as gr
import pandas as pd
from datetime import datetime
import numpy as np
from pas2 import PAS2  # Assuming the provided code is saved as pas2.py
import os

class HallucinationDetector:
    def __init__(self):
        self.pas2 = PAS2()  # Initialize with default settings
        self.results_file = "hallucination_results.xlsx"
        self._initialize_results_file()

    def _initialize_results_file(self):
        if not os.path.exists(self.results_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'original_query', 'original_response',
                'paraphrased_queries', 'paraphrased_responses',
                'hallucination_detected', 'similarity_score',
                'match_percentage', 'user_feedback'
            ])
            df.to_excel(self.results_file, index=False)

    def process_query(self, query: str):
        # Process the query using PAS2
        hallucinated, original_response, all_questions, all_responses = self.pas2.detect_hallucination(
            query, n_paraphrases=2, similarity_threshold=0.9, match_percentage_threshold=0.7
        )

        # Get embeddings and calculate matrices
        embeddings = self.pas2.get_embeddings(all_responses)
        similarity_matrix = self.pas2.calculate_similarity_matrix(embeddings)
        match_matrix = self.pas2.answer_checker(all_responses)
        match_percentage = self.pas2.calculate_match_percentage(match_matrix)

        # Format results
        results = {
            'original_query': query,
            'original_response': original_response,
            'paraphrased_queries': all_questions[1:],
            'paraphrased_responses': all_responses[1:],
            'hallucination_detected': hallucinated,
            'similarity_matrix': similarity_matrix,
            'match_percentage': match_percentage,
            'timestamp': datetime.now()
        }

        return results

    def save_feedback(self, results, feedback):
        df = pd.read_excel(self.results_file)
        new_row = {
            'timestamp': results['timestamp'],
            'original_query': results['original_query'],
            'original_response': results['original_response'],
            'paraphrased_queries': str(results['paraphrased_queries']),
            'paraphrased_responses': str(results['paraphrased_responses']),
            'hallucination_detected': results['hallucination_detected'],
            'similarity_score': np.mean(results['similarity_matrix']),
            'match_percentage': results['match_percentage'],
            'user_feedback': feedback
        }
        df = df._append(new_row, ignore_index=True)
        df.to_excel(self.results_file, index=False)

def create_interface():
    detector = HallucinationDetector()
    
    def process_and_display(query):
        results = detector.process_query(query)
        
        # Format the display outputs
        original_output = f"Query: {results['original_query']}\n\nResponse: {results['original_response']}"
        
        paraphrase_output = ""
        for i, (q, r) in enumerate(zip(results['paraphrased_queries'], results['paraphrased_responses']), 1):
            paraphrase_output += f"Paraphrase {i}:\n{q}\n\nResponse {i}:\n{r}\n\n"
        
        verdict = "🚫 Hallucination Detected" if results['hallucination_detected'] else "✅ No Hallucination Detected"
        metrics = f"""
        Similarity Score: {np.mean(results['similarity_matrix']):.3f}
        Match Percentage: {results['match_percentage']:.3f}
        """
        
        return (
            original_output,
            paraphrase_output,
            verdict,
            metrics,
            results  # Pass the full results to the feedback function
        )
    
    def submit_feedback(feedback, results):
        if results is None:
            return "Please run a query first!"
        detector.save_feedback(results, feedback)
        return "Feedback saved successfully!"

    # Create the interface
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Hallucination Detection System")
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Enter your query", lines=3)
                submit_btn = gr.Button("Submit", variant="primary")

        with gr.Row():
            with gr.Column():
                original_output = gr.Textbox(label="Original Query and Response", lines=6)
            with gr.Column():
                paraphrase_output = gr.Textbox(label="Paraphrased Queries and Responses", lines=6)

        with gr.Row():
            with gr.Column():
                verdict_output = gr.Textbox(label="Verdict")
            with gr.Column():
                metrics_output = gr.Textbox(label="Metrics")

        with gr.Row():
            with gr.Column():
                feedback = gr.Radio(
                    choices=["Correct", "Incorrect", "Unsure"],
                    label="Was the hallucination detection correct?"
                )
                feedback_btn = gr.Button("Submit Feedback")
                feedback_output = gr.Textbox(label="Feedback Status")

        # Hidden state for storing results
        results_state = gr.State()

        # Set up event handlers
        submit_btn.click(
            process_and_display,
            inputs=[query_input],
            outputs=[original_output, paraphrase_output, verdict_output, metrics_output, results_state]
        )

        feedback_btn.click(
            submit_feedback,
            inputs=[feedback, results_state],
            outputs=[feedback_output]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()