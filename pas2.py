import os
import gradio as gr
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from mistralai import Mistral
from openai import OpenAI
import re
import json
import logging
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class HallucinationJudgment(BaseModel):
    hallucination_detected: bool = Field(description="Whether a hallucination is detected across the responses")
    confidence_score: float = Field(description="Confidence score between 0-1 for the hallucination judgment")
    conflicting_facts: List[Dict[str, Any]] = Field(description="List of conflicting facts found in the responses")
    reasoning: str = Field(description="Detailed reasoning for the judgment")
    summary: str = Field(description="A summary of the analysis")

class PAS2:
    """Paraphrase-based Approach for Scrutinizing Systems - Using model-as-judge"""
    
    def __init__(self, mistral_api_key=None, openai_api_key=None, progress_callback=None):
        """Initialize the PAS2 with API keys"""
        # For Hugging Face Spaces, we prioritize getting API keys from HF_* environment variables
        # which are set from the Secrets tab in the Space settings
        self.mistral_api_key = mistral_api_key or os.environ.get("HF_MISTRAL_API_KEY") or os.environ.get("MISTRAL_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("HF_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.progress_callback = progress_callback
        
        if not self.mistral_api_key:
            raise ValueError("Mistral API key is required. Set it via HF_MISTRAL_API_KEY in Hugging Face Spaces secrets or pass it as a parameter.")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set it via HF_OPENAI_API_KEY in Hugging Face Spaces secrets or pass it as a parameter.")
        
        self.mistral_client = Mistral(api_key=self.mistral_api_key)
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        self.mistral_model = "mistral-large-latest"
        self.openai_model = "o3-mini"
        
        logger.info("PAS2 initialized with Mistral model: %s and OpenAI model: %s", 
                   self.mistral_model, self.openai_model)
    
    def generate_paraphrases(self, query: str, n_paraphrases: int = 3) -> List[str]:
        """Generate paraphrases of the input query using Mistral API"""
        logger.info("Generating %d paraphrases for query: %s", n_paraphrases, query)
        start_time = time.time()
        
        messages = [
            {
                "role": "system",
                "content": f"You are an expert at creating semantically equivalent paraphrases. Generate {n_paraphrases} different paraphrases of the given query that preserve the original meaning but vary in wording and structure. Return a JSON array of strings, each containing one paraphrase."
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        try:
            logger.info("Sending paraphrase generation request to Mistral API...")
            response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            logger.debug("Received raw paraphrase response: %s", content)
            
            paraphrases_data = json.loads(content)
            
            # Handle different possible JSON structures
            if isinstance(paraphrases_data, dict) and "paraphrases" in paraphrases_data:
                paraphrases = paraphrases_data["paraphrases"]
            elif isinstance(paraphrases_data, dict) and "results" in paraphrases_data:
                paraphrases = paraphrases_data["results"]
            elif isinstance(paraphrases_data, list):
                paraphrases = paraphrases_data
            else:
                # Try to extract a list from any field
                for key, value in paraphrases_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        paraphrases = value
                        break
                else:
                    logger.warning("Could not extract paraphrases from response: %s", content)
                    raise ValueError(f"Could not extract paraphrases from response: {content}")
            
            # Ensure we have the right number of paraphrases
            paraphrases = paraphrases[:n_paraphrases]
            
            # Add the original query as the first item
            all_queries = [query] + paraphrases
            
            elapsed_time = time.time() - start_time
            logger.info("Generated %d paraphrases in %.2f seconds", len(paraphrases), elapsed_time)
            for i, p in enumerate(paraphrases, 1):
                logger.info("Paraphrase %d: %s", i, p)
            
            return all_queries
            
        except Exception as e:
            logger.error("Error generating paraphrases: %s", str(e), exc_info=True)
            # Return original plus simple paraphrases as fallback
            fallback_paraphrases = [
                query,
                f"Could you tell me about {query.strip('?')}?",
                f"I'd like to know: {query}",
                f"Please provide information on {query.strip('?')}."
            ][:n_paraphrases+1]
            
            logger.info("Using fallback paraphrases due to error")
            for i, p in enumerate(fallback_paraphrases[1:], 1):
                logger.info("Fallback paraphrase %d: %s", i, p)
                
            return fallback_paraphrases
    
    def _get_single_response(self, query: str, index: int = None) -> str:
        """Get a single response from Mistral API for a query"""
        try:
            query_description = f"Query {index}: {query}" if index is not None else f"Query: {query}"
            logger.info("Getting response for %s", query_description)
            start_time = time.time()
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Provide accurate, factual information in response to questions."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            
            response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=messages
            )
            
            result = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            
            logger.info("Received response for %s (%.2f seconds)", query_description, elapsed_time)
            logger.debug("Response content for %s: %s", query_description, result[:100] + "..." if len(result) > 100 else result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error getting response for query '{query}': {e}"
            logger.error(error_msg, exc_info=True)
            return f"Error: Failed to get response for this query."
    
    def get_responses(self, queries: List[str]) -> List[str]:
        """Get responses from Mistral API for each query in parallel"""
        logger.info("Getting responses for %d queries in parallel", len(queries))
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
            # Submit tasks and map them to their original indices
            future_to_index = {
                executor.submit(self._get_single_response, query, i): i 
                for i, query in enumerate(queries)
            }
            
            # Prepare a list with the correct length
            responses = [""] * len(queries)
            
            # Counter for completed responses
            completed_count = 0
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    responses[index] = future.result()
                    
                    # Update completion count and report progress
                    completed_count += 1
                    if self.progress_callback:
                        self.progress_callback("responses_progress", 
                                            completed_responses=completed_count, 
                                            total_responses=len(queries))
                        
                except Exception as e:
                    logger.error("Error processing response for index %d: %s", index, str(e))
                    responses[index] = f"Error: Failed to get response for query {index}."
                    
                    # Still update completion count even for errors
                    completed_count += 1
                    if self.progress_callback:
                        self.progress_callback("responses_progress", 
                                            completed_responses=completed_count, 
                                            total_responses=len(queries))
        
        elapsed_time = time.time() - start_time
        logger.info("Received all %d responses in %.2f seconds total", len(responses), elapsed_time)
        
        return responses
    
    def detect_hallucination(self, query: str, n_paraphrases: int = 3) -> Dict:
        """
        Detect hallucinations by comparing responses to paraphrased queries using a judge model
        
        Returns:
            Dict containing hallucination judgment and all responses
        """
        logger.info("Starting hallucination detection for query: %s", query)
        start_time = time.time()
        
        # Report progress
        if self.progress_callback:
            self.progress_callback("starting", query=query)
            
        # Generate paraphrases
        logger.info("Step 1: Generating paraphrases")
        if self.progress_callback:
            self.progress_callback("generating_paraphrases", query=query)
            
        all_queries = self.generate_paraphrases(query, n_paraphrases)
        
        if self.progress_callback:
            self.progress_callback("paraphrases_complete", query=query, count=len(all_queries))
        
        # Get responses to all queries
        logger.info("Step 2: Getting responses to all %d queries", len(all_queries))
        if self.progress_callback:
            self.progress_callback("getting_responses", query=query, total=len(all_queries))
        
        all_responses = []
        for i, q in enumerate(all_queries):
            logger.info("Getting response %d/%d for query: %s", i+1, len(all_queries), q)
            if self.progress_callback:
                self.progress_callback("responses_progress", query=query, completed=i, total=len(all_queries))
            
            response = self._get_single_response(q, index=i)
            all_responses.append(response)
        
        if self.progress_callback:
            self.progress_callback("responses_complete", query=query)
        
        # Judge the responses for hallucinations
        logger.info("Step 3: Judging for hallucinations")
        if self.progress_callback:
            self.progress_callback("judging", query=query)
        
        # The first query is the original, rest are paraphrases
        original_query = all_queries[0]
        original_response = all_responses[0]
        paraphrased_queries = all_queries[1:] if len(all_queries) > 1 else []
        paraphrased_responses = all_responses[1:] if len(all_responses) > 1 else []
        
        # Judge the responses
        judgment = self.judge_hallucination(
            original_query=original_query,
            original_response=original_response,
            paraphrased_queries=paraphrased_queries,
            paraphrased_responses=paraphrased_responses
        )
        
        # Assemble the results
        results = {
            "original_query": original_query,
            "original_response": original_response,
            "paraphrased_queries": paraphrased_queries,
            "paraphrased_responses": paraphrased_responses,
            "hallucination_detected": judgment.hallucination_detected,
            "confidence_score": judgment.confidence_score,
            "conflicting_facts": judgment.conflicting_facts,
            "reasoning": judgment.reasoning,
            "summary": judgment.summary
        }
        
        # Report completion
        if self.progress_callback:
            self.progress_callback("complete", query=query)
            
        logger.info("Hallucination detection completed in %.2f seconds", time.time() - start_time)
        return results
    
    def judge_hallucination(self, 
                           original_query: str, 
                           original_response: str, 
                           paraphrased_queries: List[str], 
                           paraphrased_responses: List[str]) -> HallucinationJudgment:
        """
        Use OpenAI's o3-mini as a judge to detect hallucinations in the responses
        """
        logger.info("Judging hallucinations with OpenAI's %s model", self.openai_model)
        start_time = time.time()
        
        # Prepare the context for the judge
        context = f"""
Original Question: {original_query}

Original Response: 
{original_response}

Paraphrased Questions and their Responses:
"""
        
        for i, (query, response) in enumerate(zip(paraphrased_queries, paraphrased_responses), 1):
            context += f"\nParaphrased Question {i}: {query}\n\nResponse {i}:\n{response}\n"
        
        system_prompt = """
You are a judge evaluating whether an AI is hallucinating across different responses to semantically equivalent questions.
Analyze all responses carefully to identify any factual inconsistencies or contradictions.
Focus on factual discrepancies, not stylistic differences.
A hallucination is when the AI states different facts in response to questions that are asking for the same information.

Your response should be a JSON with the following fields:
- hallucination_detected: boolean indicating whether hallucinations were found
- confidence_score: number between 0 and 1 representing your confidence in the judgment
- conflicting_facts: an array of objects describing any conflicting information found
- reasoning: detailed explanation for your judgment
- summary: a concise summary of your analysis
"""

        try:
            logger.info("Sending judgment request to OpenAI API...")
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Evaluate these responses for hallucinations:\n\n{context}"}
                ],
                response_format={"type": "json_object"}
            )
            
            result_json = json.loads(response.choices[0].message.content)
            logger.debug("Received judgment response: %s", result_json)
            
            # Create the HallucinationJudgment object from the JSON response
            judgment = HallucinationJudgment(
                hallucination_detected=result_json.get("hallucination_detected", False),
                confidence_score=result_json.get("confidence_score", 0.0),
                conflicting_facts=result_json.get("conflicting_facts", []),
                reasoning=result_json.get("reasoning", "No reasoning provided."),
                summary=result_json.get("summary", "No summary provided.")
            )
            
            elapsed_time = time.time() - start_time
            logger.info("Judgment completed in %.2f seconds", elapsed_time)
            
            return judgment
            
        except Exception as e:
            logger.error("Error in hallucination judgment: %s", str(e), exc_info=True)
            # Return a fallback judgment
            return HallucinationJudgment(
                hallucination_detected=False,
                confidence_score=0.0,
                conflicting_facts=[],
                reasoning="Failed to obtain judgment from the model.",
                summary="Analysis failed due to API error."
            )


class HallucinationDetectorApp:
    def __init__(self):
        self.pas2 = None
        self.results_file = "hallucination_results.xlsx"
        logger.info("Initializing HallucinationDetectorApp")
        self._initialize_results_file()
        self.progress_callback = None
    
    def _initialize_results_file(self):
        if not os.path.exists(self.results_file):
            logger.info("Creating new results file: %s", self.results_file)
            df = pd.DataFrame(columns=[
                'timestamp', 'original_query', 'original_response',
                'paraphrased_queries', 'paraphrased_responses',
                'hallucination_detected', 'confidence_score', 
                'conflicting_facts', 'reasoning', 'summary', 'user_feedback'
            ])
            df.to_excel(self.results_file, index=False)
        else:
            logger.info("Results file already exists: %s", self.results_file)
    
    def set_progress_callback(self, callback):
        """Set the progress callback function"""
        self.progress_callback = callback
    
    def initialize_api(self, mistral_api_key, openai_api_key):
        """Initialize the PAS2 with API keys"""
        try:
            logger.info("Initializing PAS2 with API keys")
            self.pas2 = PAS2(
                mistral_api_key=mistral_api_key, 
                openai_api_key=openai_api_key,
                progress_callback=self.progress_callback
            )
            logger.info("API initialization successful")
            return "API keys set successfully! You can now use the application."
        except Exception as e:
            logger.error("Error initializing API: %s", str(e), exc_info=True)
            return f"Error initializing API: {str(e)}"
    
    def process_query(self, query: str):
        """Process the query using PAS2"""
        if not self.pas2:
            logger.error("PAS2 not initialized")
            return {
                "error": "Please set API keys first before processing queries."
            }
        
        if not query.strip():
            logger.warning("Empty query provided")
            return {
                "error": "Please enter a query."
            }
        
        try:
            # Set the progress callback if needed
            if self.progress_callback and self.pas2.progress_callback != self.progress_callback:
                self.pas2.progress_callback = self.progress_callback
                
            # Process the query
            logger.info("Processing query with PAS2: %s", query)
            results = self.pas2.detect_hallucination(query)
            logger.info("Query processing completed successfully")
            return results
        except Exception as e:
            logger.error("Error processing query: %s", str(e), exc_info=True)
            return {
                "error": f"Error processing query: {str(e)}"
            }
    
    def save_feedback(self, results, feedback):
        """Save results and user feedback to Excel file"""
        try:
            logger.info("Saving user feedback: %s", feedback)
            # Read existing data
            if os.path.exists(self.results_file):
                logger.debug("Reading existing results file")
                df = pd.read_excel(self.results_file)
            else:
                logger.debug("Creating new results DataFrame")
                df = pd.DataFrame(columns=[
                    'timestamp', 'original_query', 'original_response',
                    'paraphrased_queries', 'paraphrased_responses',
                    'hallucination_detected', 'confidence_score', 
                    'conflicting_facts', 'reasoning', 'summary', 'user_feedback'
                ])
            
            # Prepare data to save
            data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'original_query': results.get('original_query', ''),
                'original_response': results.get('original_response', ''),
                'paraphrased_queries': str(results.get('paraphrased_queries', [])),
                'paraphrased_responses': str(results.get('paraphrased_responses', [])),
                'hallucination_detected': results.get('hallucination_detected', False),
                'confidence_score': results.get('confidence_score', 0.0),
                'conflicting_facts': str(results.get('conflicting_facts', [])),
                'reasoning': results.get('reasoning', ''),
                'summary': results.get('summary', ''),
                'user_feedback': feedback
            }
            
            # Append new data
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            
            # Save to Excel
            logger.debug("Saving feedback to Excel file")
            df.to_excel(self.results_file, index=False)
            
            logger.info("Feedback saved successfully")
            return "Feedback saved successfully!"
        except Exception as e:
            logger.error("Error saving feedback: %s", str(e), exc_info=True)
            return f"Error saving feedback: {str(e)}"


# Progress tracking for UI updates
class ProgressTracker:
    """Tracks progress of hallucination detection for UI updates"""
    
    STAGES = {
        "idle": {"status": "Ready", "progress": 0, "color": "#757575"},
        "starting": {"status": "Starting process...", "progress": 5, "color": "#2196F3"},
        "generating_paraphrases": {"status": "Generating paraphrases...", "progress": 15, "color": "#2196F3"},
        "paraphrases_complete": {"status": "Paraphrases generated", "progress": 30, "color": "#2196F3"},
        "getting_responses": {"status": "Getting responses (0/0)...", "progress": 35, "color": "#2196F3"},
        "responses_progress": {"status": "Getting responses ({completed}/{total})...", "progress": 40, "color": "#2196F3"},
        "responses_complete": {"status": "All responses received", "progress": 65, "color": "#2196F3"},
        "judging": {"status": "Analyzing responses for hallucinations...", "progress": 70, "color": "#2196F3"},
        "complete": {"status": "Analysis complete!", "progress": 100, "color": "#4CAF50"},
        "error": {"status": "Error: {error_message}", "progress": 100, "color": "#F44336"}
    }
    
    def __init__(self):
        self.stage = "idle"
        self.stage_data = self.STAGES[self.stage].copy()
        self.query = ""
        self.completed_responses = 0
        self.total_responses = 0
        self.error_message = ""
        self._lock = threading.Lock()
        self._status_callback = None
        self._stop_event = threading.Event()
        self._update_thread = None
    
    def register_callback(self, callback_fn):
        """Register callback function to update UI"""
        self._status_callback = callback_fn
    
    def update_stage(self, stage, **kwargs):
        """Update the current stage and trigger callback"""
        with self._lock:
            if stage in self.STAGES:
                self.stage = stage
                self.stage_data = self.STAGES[stage].copy()
                
                # Update with any additional parameters
                for key, value in kwargs.items():
                    if key == 'query':
                        self.query = value
                    elif key == 'completed_responses':
                        self.completed_responses = value
                    elif key == 'total_responses':
                        self.total_responses = value
                    elif key == 'error_message':
                        self.error_message = value
                
                # Format status message
                if stage == 'responses_progress':
                    self.stage_data['status'] = self.stage_data['status'].format(
                        completed=self.completed_responses, 
                        total=self.total_responses
                    )
                elif stage == 'error':
                    self.stage_data['status'] = self.stage_data['status'].format(
                        error_message=self.error_message
                    )
                
                if self._status_callback:
                    self._status_callback(self.get_html_status())
    
    def get_html_status(self):
        """Get HTML representation of current status"""
        progress_width = f"{self.stage_data['progress']}%"
        status_text = self.stage_data['status']
        color = self.stage_data['color']
        
        query_info = f'<div class="query-display">{self.query}</div>' if self.query else ''
        
        # Only show status text if not in idle state
        status_display = f'<div class="progress-status" style="color: {color};">{status_text}</div>' if self.stage != "idle" else ''
        
        html = f"""
        <div class="progress-container">
            {query_info}
            {status_display}
            <div class="progress-bar-container">
                <div class="progress-bar" style="width: {progress_width}; background-color: {color};"></div>
            </div>
        </div>
        """
        return html
    
    def start_pulsing(self):
        """Start a pulsing animation for the progress bar during long operations"""
        if self._update_thread and self._update_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._pulse_progress)
        self._update_thread.daemon = True
        self._update_thread.start()
    
    def stop_pulsing(self):
        """Stop the pulsing animation"""
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(0.5)
    
    def _pulse_progress(self):
        """Animate the progress bar to show activity"""
        pulse_stages = ["⋯", "⋯⋯", "⋯⋯⋯", "⋯⋯", "⋯"]
        i = 0
        while not self._stop_event.is_set():
            with self._lock:
                if self.stage not in ["idle", "complete", "error"]:
                    status_base = self.stage_data['status'].split("...")[0] if "..." in self.stage_data['status'] else self.stage_data['status']
                    self.stage_data['status'] = f"{status_base}... {pulse_stages[i]}"
                    
                    if self._status_callback:
                        self._status_callback(self.get_html_status())
            
            i = (i + 1) % len(pulse_stages)
            time.sleep(0.3)


def create_interface():
    """Create Gradio interface"""
    detector = HallucinationDetectorApp()
    
    # Initialize Progress Tracker
    progress_tracker = ProgressTracker()
    
    # Initialize APIs from environment variables automatically
    try:
        detector.initialize_api(
            mistral_api_key=os.environ.get("HF_MISTRAL_API_KEY"),
            openai_api_key=os.environ.get("HF_OPENAI_API_KEY")
        )
    except Exception as e:
        print(f"Warning: Failed to initialize APIs from environment variables: {e}")
        print("Please make sure HF_MISTRAL_API_KEY and HF_OPENAI_API_KEY are set in your environment")
    
    # CSS for styling
    css = """
    .container {
        max-width: 1000px;
        margin: 0 auto;
    }
    .title {
        text-align: center;
        margin-bottom: 0.5em;
        color: #1a237e;
        font-weight: 600;
    }
    .subtitle {
        text-align: center;
        margin-bottom: 1.5em;
        color: #455a64;
        font-size: 1.2em;
    }
    .section-title {
        margin-top: 1em;
        margin-bottom: 0.5em;
        font-weight: bold;
        color: #283593;
    }
    .info-box {
        padding: 1.2em;
        border-radius: 8px;
        background-color: #f5f5f5;
        margin-bottom: 1em;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .hallucination-positive {
        padding: 1.2em;
        border-radius: 8px;
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        margin-bottom: 1em;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .hallucination-negative {
        padding: 1.2em;
        border-radius: 8px;
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        margin-bottom: 1em;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .response-box {
        padding: 1.2em;
        border-radius: 8px;
        background-color: #f5f5f5;
        margin-bottom: 0.8em;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .example-queries {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 15px;
    }
    .example-query {
        background-color: #e3f2fd;
        padding: 8px 15px;
        border-radius: 18px;
        font-size: 0.9em;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid #bbdefb;
    }
    .example-query:hover {
        background-color: #bbdefb;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stats-section {
        display: flex;
        justify-content: space-between;
        background-color: #e8eaf6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .stat-item {
        text-align: center;
        padding: 10px;
    }
    .stat-value {
        font-size: 1.5em;
        font-weight: bold;
        color: #303f9f;
    }
    .stat-label {
        font-size: 0.9em;
        color: #5c6bc0;
    }
    .feedback-section {
        border-top: 1px solid #e0e0e0;
        padding-top: 15px;
        margin-top: 20px;
    }
    footer {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        color: #9e9e9e;
        font-size: 0.9em;
    }
    .processing-status {
        padding: 12px;
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        margin-bottom: 15px;
        font-weight: 500;
        color: #e65100;
    }
    .debug-panel {
        background-color: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 10px;
        margin-top: 15px;
        font-family: monospace;
        font-size: 0.9em;
        white-space: pre-wrap;
        max-height: 200px;
        overflow-y: auto;
    }
    .progress-container {
        padding: 15px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .progress-status {
        font-weight: 500;
        margin-bottom: 8px;
        padding: 4px 0;
        font-size: 0.95em;
    }
    .progress-bar-container {
        background-color: #e0e0e0;
        height: 10px;
        border-radius: 5px;
        overflow: hidden;
        margin-bottom: 10px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .progress-bar {
        height: 100%;
        transition: width 0.5s ease;
        background-image: linear-gradient(to right, #2196F3, #3f51b5);
    }
    .query-display {
        font-style: italic;
        color: #666;
        margin-bottom: 10px;
        background-color: #f5f5f5;
        padding: 8px;
        border-radius: 4px;
        border-left: 3px solid #2196F3;
    }
    """
    
    # Example queries
    example_queries = [
        "Who was the first person to land on the moon?",
        "What is the capital of France?",
        "How many planets are in our solar system?",
        "Who wrote the novel 1984?",
        "What is the speed of light?",
        "What was the first computer?"
    ]
    
    # Function to update the progress display
    def update_progress_display(html):
        """Update the progress display with the provided HTML"""
        return gr.update(visible=True, value=html)
    
    # Register the callback with the tracker
    progress_tracker.register_callback(update_progress_display)
    
    # Register the tracker with the detector
    detector.set_progress_callback(progress_tracker.update_stage)
    
    # Helper function to set example query
    def set_example_query(example):
        return example
    
    # Function to show processing is starting
    def start_processing(query):
        logger.info("Processing query: %s", query)
        # Stop any existing pulsing to prepare for incremental progress updates
        progress_tracker.stop_pulsing()
        
        # Reset to a processing state without the "Ready" text
        # Use "starting" stage but with minimal UI display
        progress_tracker.stage = "starting"
        progress_tracker.query = query
        
        # Force UI update with clean display
        if progress_tracker._status_callback:
            progress_tracker._status_callback(progress_tracker.get_html_status())
        
        return [
            gr.update(visible=True),  # Show the progress display
            gr.update(visible=False),  # Hide the results accordion
            gr.update(visible=False),  # Hide the feedback accordion
            None  # Reset hidden results
        ]
    
    # Main processing function
    def process_query_and_display_results(query, progress=gr.Progress()):
        if not query.strip():
            logger.warning("Empty query submitted")
            progress_tracker.stop_pulsing()
            progress_tracker.update_stage("error", error_message="Please enter a query.")
            return [
                gr.update(visible=True),  # Show the progress with error
                gr.update(visible=False),
                gr.update(visible=False),
                None
            ]
            
        # Check if API is initialized
        if not detector.pas2:
            try:
                # Try to initialize from environment variables
                logger.info("Initializing APIs from environment variables")
                progress(0.05, desc="Initializing API...")
                init_message = detector.initialize_api(
                    mistral_api_key=os.environ.get("HF_MISTRAL_API_KEY"),
                    openai_api_key=os.environ.get("HF_OPENAI_API_KEY")
                )
                if "successfully" not in init_message:
                    logger.error("Failed to initialize APIs: %s", init_message)
                    progress_tracker.stop_pulsing()
                    progress_tracker.update_stage("error", error_message="API keys not found in environment variables.")
                    return [
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None
                    ]
            except Exception as e:
                logger.error("Error initializing API: %s", str(e), exc_info=True)
                progress_tracker.stop_pulsing()
                progress_tracker.update_stage("error", error_message=f"Error initializing API: {str(e)}")
                return [
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    None
                ]
        
        try:
            # Process the query
            logger.info("Starting hallucination detection process")
            start_time = time.time()
            
            # Set up a custom progress callback that uses both the progress_tracker and the gr.Progress
            def combined_progress_callback(stage, **kwargs):
                # Skip the idle stage, which shows "Ready"
                if stage == "idle":
                    return
                    
                progress_tracker.update_stage(stage, **kwargs)
                
                # Map the stages to progress values for the gr.Progress bar
                stage_to_progress = {
                    "starting": 0.05,
                    "generating_paraphrases": 0.15,
                    "paraphrases_complete": 0.3,
                    "getting_responses": 0.35,
                    "responses_progress": lambda kwargs: 0.35 + (0.3 * (kwargs.get("completed", 0) / max(kwargs.get("total", 1), 1))),
                    "responses_complete": 0.65,
                    "judging": 0.7,
                    "complete": 1.0,
                    "error": 1.0
                }
                
                # Update the gr.Progress bar
                if stage in stage_to_progress:
                    prog_value = stage_to_progress[stage]
                    if callable(prog_value):
                        prog_value = prog_value(kwargs)
                    
                    desc = progress_tracker.STAGES[stage]["status"]
                    if "{" in desc and "}" in desc:
                        # Format the description with any kwargs
                        desc = desc.format(**kwargs)
                    
                    # Ensure UI updates by adding a small delay
                    # This forces the progress updates to be rendered
                    progress(prog_value, desc=desc)
                    
                    # For certain key stages, add a small sleep to ensure progress is visible
                    if stage in ["starting", "generating_paraphrases", "paraphrases_complete", 
                                "getting_responses", "responses_complete", "judging", "complete"]:
                        time.sleep(0.2)  # Small delay to ensure UI update is visible
            
            # Use these steps for processing
            detector.set_progress_callback(combined_progress_callback)
            
            # Create a wrapper function for detect_hallucination that gives more control over progress updates
            def run_detection_with_visible_progress():
                # Step 1: Start
                combined_progress_callback("starting", query=query)
                time.sleep(0.3)  # Ensure starting status is visible
                
                # Step 2: Generate paraphrases (15-30%)
                combined_progress_callback("generating_paraphrases", query=query)
                all_queries = detector.pas2.generate_paraphrases(query)
                combined_progress_callback("paraphrases_complete", query=query, count=len(all_queries))
                
                # Step 3: Get responses (35-65%)
                combined_progress_callback("getting_responses", query=query, total=len(all_queries))
                all_responses = []
                for i, q in enumerate(all_queries):
                    # Show incremental progress for each response
                    combined_progress_callback("responses_progress", query=query, completed=i, total=len(all_queries))
                    response = detector.pas2._get_single_response(q, index=i)
                    all_responses.append(response)
                combined_progress_callback("responses_complete", query=query)
                
                # Step 4: Judge hallucinations (70-100%)
                combined_progress_callback("judging", query=query)
                
                # The first query is the original, rest are paraphrases
                original_query = all_queries[0]
                original_response = all_responses[0]
                paraphrased_queries = all_queries[1:] if len(all_queries) > 1 else []
                paraphrased_responses = all_responses[1:] if len(all_responses) > 1 else []
                
                # Judge the responses
                judgment = detector.pas2.judge_hallucination(
                    original_query=original_query,
                    original_response=original_response,
                    paraphrased_queries=paraphrased_queries,
                    paraphrased_responses=paraphrased_responses
                )
                
                # Assemble the results
                results = {
                    "original_query": original_query,
                    "original_response": original_response,
                    "paraphrased_queries": paraphrased_queries,
                    "paraphrased_responses": paraphrased_responses,
                    "hallucination_detected": judgment.hallucination_detected,
                    "confidence_score": judgment.confidence_score,
                    "conflicting_facts": judgment.conflicting_facts,
                    "reasoning": judgment.reasoning,
                    "summary": judgment.summary
                }
                
                # Show completion
                combined_progress_callback("complete", query=query)
                time.sleep(0.3)  # Ensure complete status is visible
                
                return results
            
            # Run the detection process with visible progress
            results = run_detection_with_visible_progress()
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logger.info("Hallucination detection completed in %.2f seconds", elapsed_time)
            
            # Check for errors
            if "error" in results:
                logger.error("Error in results: %s", results["error"])
                progress_tracker.stop_pulsing()
                progress_tracker.update_stage("error", error_message=results["error"])
                return [
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    None
                ]
            
            # Prepare responses for display
            original_query = results["original_query"]
            original_response = results["original_response"]
            
            paraphrased_queries = results["paraphrased_queries"]
            paraphrased_responses = results["paraphrased_responses"]
            
            hallucination_detected = results["hallucination_detected"]
            confidence = results["confidence_score"]
            reasoning = results["reasoning"]
            summary = results["summary"]
            
            # Format conflicting facts
            conflicting_facts = results["conflicting_facts"]
            conflicting_facts_text = ""
            if conflicting_facts:
                for i, fact in enumerate(conflicting_facts, 1):
                    conflicting_facts_text += f"{i}. "
                    if isinstance(fact, dict):
                        for key, value in fact.items():
                            conflicting_facts_text += f"{key}: {value}, "
                        conflicting_facts_text = conflicting_facts_text.rstrip(", ")
                    else:
                        conflicting_facts_text += str(fact)
                    conflicting_facts_text += "\n"
            
            # Create HTML display
            html_output = f"""
            <div class="container">
                <h2 class="title">Hallucination Detection Results</h2>
                
                <div class="stats-section">
                    <div class="stat-item">
                        <div class="stat-value">{'Yes' if hallucination_detected else 'No'}</div>
                        <div class="stat-label">Hallucination Detected</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{confidence:.2f}</div>
                        <div class="stat-label">Confidence Score</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{len(paraphrased_queries)}</div>
                        <div class="stat-label">Paraphrases Analyzed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{elapsed_time:.1f}s</div>
                        <div class="stat-label">Processing Time</div>
                    </div>
                </div>
                
                <div class="{'hallucination-positive' if hallucination_detected else 'hallucination-negative'}">
                    <h3>Analysis Summary</h3>
                    <p>{summary}</p>
                </div>
                
                <div class="section-title">Original Query</div>
                <div class="response-box">
                    {original_query}
                </div>
                
                <div class="section-title">Original Response</div>
                <div class="response-box">
                    {original_response.replace('\n', '<br>')}
                </div>
                
                <div class="section-title">Paraphrased Queries and Responses</div>
            """
            
            for i, (q, r) in enumerate(zip(paraphrased_queries, paraphrased_responses), 1):
                html_output += f"""
                <div class="section-title">Paraphrased Query {i}</div>
                <div class="response-box">
                    {q}
                </div>
                
                <div class="section-title">Response {i}</div>
                <div class="response-box">
                    {r.replace('\n', '<br>')}
                </div>
                """
            
            html_output += f"""
                <div class="section-title">Detailed Analysis</div>
                <div class="info-box">
                    <p><strong>Reasoning:</strong></p>
                    <p>{reasoning.replace('\n', '<br>')}</p>
                    
                    <p><strong>Conflicting Facts:</strong></p>
                    <p>{conflicting_facts_text.replace('\n', '<br>') if conflicting_facts_text else "None identified"}</p>
                </div>
            </div>
            """
            
            logger.info("Updating UI with results")
            progress_tracker.stop_pulsing()
            
            return [
                gr.update(visible=False),  # Hide progress display when showing results
                gr.update(visible=True, value=html_output),
                gr.update(visible=True),
                results
            ]
            
        except Exception as e:
            logger.error("Error processing query: %s", str(e), exc_info=True)
            progress_tracker.stop_pulsing()
            progress_tracker.update_stage("error", error_message=f"Error processing query: {str(e)}")
            return [
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                None
            ]
    
    # Helper function to submit feedback
    def combine_feedback(fb_input, fb_text, results):
        combined_feedback = f"{fb_input}: {fb_text}" if fb_text else fb_input
        if not results:
            return "No results to attach feedback to."
        
        response = detector.save_feedback(results, combined_feedback)
        return response
    
    # Create the interface
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as interface:
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 1.5rem">
                <h1 style="font-size: 2.2em; font-weight: 600; color: #1a237e; margin-bottom: 0.2em;">PAS2 - Hallucination Detector</h1>
                <h3 style="font-size: 1.3em; color: #455a64; margin-bottom: 0.8em;">Advanced AI Response Verification Using Model-as-Judge</h3>
                <p style="font-size: 1.1em; color: #546e7a; max-width: 800px; margin: 0 auto;">
                    This tool detects hallucinations in AI responses by comparing answers to semantically equivalent questions and using a specialized judge model.
                </p>
            </div>
            """
        )
        
        with gr.Accordion("About this Tool", open=False):
            gr.Markdown(
                """
                ### How It Works
                
                This tool implements the Paraphrase-based Approach for Scrutinizing Systems (PAS2) with a model-as-judge enhancement:
                
                1. **Paraphrase Generation**: Your question is paraphrased multiple ways while preserving its core meaning
                2. **Multiple Responses**: All questions (original + paraphrases) are sent to Mistral Large model
                3. **Expert Judgment**: OpenAI's o3-mini analyzes all responses to detect factual inconsistencies
                
                ### Why This Approach?
                
                When an AI hallucinates, it often provides different answers to the same question when phrased differently. 
                By using a separate judge model, we can identify these inconsistencies more effectively than with 
                metric-based approaches.
                
                ### Understanding the Results
                
                - **Confidence Score**: Indicates the judge's confidence in the hallucination detection
                - **Conflicting Facts**: Specific inconsistencies found across responses
                - **Reasoning**: The judge's detailed analysis explaining its decision
                
                ### Privacy Notice
                
                Your queries and the system's responses are saved to help improve hallucination detection.
                No personally identifiable information is collected.
                """
            )
        
        with gr.Row():
            with gr.Column():
                # First define the query input
                gr.Markdown("### Enter Your Question")
                with gr.Row():
                    query_input = gr.Textbox(
                        label="",
                        placeholder="Ask a factual question (e.g., Who was the first person to land on the moon?)",
                        lines=3
                    )
                
                # Now define the example queries
                gr.Markdown("### Or Try an Example")
                example_row = gr.Row()
                with example_row:
                    for example in example_queries:
                        example_btn = gr.Button(
                            example, 
                            elem_classes=["example-query"],
                            scale=0
                        )
                        example_btn.click(
                            fn=set_example_query,
                            inputs=[gr.Textbox(value=example, visible=False)],
                            outputs=[query_input]
                        )
                
                with gr.Row():
                    submit_button = gr.Button("Detect Hallucinations", variant="primary", scale=1)
        
        # Error message
        error_message = gr.HTML(
            label="Status",
            visible=False
        )
        
        # Progress display
        progress_display = gr.HTML(
            value=progress_tracker.get_html_status(),
            visible=True
        )
        
        # Results display
        results_accordion = gr.HTML(visible=False)
        
        # Feedback section
        with gr.Accordion("Provide Feedback", open=False, visible=False) as feedback_accordion:
            gr.Markdown("### Help Improve the System")
            gr.Markdown("Your feedback helps us refine the hallucination detection system.")
            
            feedback_input = gr.Radio(
                label="Is the hallucination detection accurate?",
                choices=["Yes, correct detection", "No, incorrectly flagged hallucination", "No, missed hallucination", "Unsure/Other"],
                value="Yes, correct detection"
            )
            
            feedback_text = gr.Textbox(
                label="Additional comments (optional)",
                placeholder="Please provide any additional observations or details...",
                lines=2
            )
            
            feedback_button = gr.Button("Submit Feedback", variant="secondary")
            feedback_status = gr.Textbox(label="Feedback Status", interactive=False, visible=False)
            
            feedback_button.click(
                fn=lambda: gr.update(visible=True),
                outputs=[feedback_status]
            )
        
        # Hidden state to store results for feedback
        hidden_results = gr.State()
        
        # Set up event handlers
        submit_button.click(
            fn=start_processing,
            inputs=[query_input],
            outputs=[progress_display, results_accordion, feedback_accordion, hidden_results],
            queue=False
        ).then(
            fn=process_query_and_display_results,
            inputs=[query_input],
            outputs=[progress_display, results_accordion, feedback_accordion, hidden_results]
        )
        
        feedback_button.click(
            fn=combine_feedback,
            inputs=[feedback_input, feedback_text, hidden_results],
            outputs=[feedback_status]
        )
        
        # Footer
        gr.HTML(
            """
            <footer>
                <p>Paraphrase-based Approach for Scrutinizing Systems (PAS2) - Advanced Hallucination Detection</p>
                <p>Using Mistral Large for generation and OpenAI o3-mini as judge</p>
            </footer>
            """
        )
    
    return interface

# Add a test function to demonstrate progress bar in isolation
def test_progress():
    """Simple test function to demonstrate progress bar"""
    import gradio as gr
    import time
    
    def slow_process(progress=gr.Progress()):
        progress(0, desc="Starting process...")
        time.sleep(0.5)
        
        # Phase 1: Generating paraphrases
        progress(0.15, desc="Generating paraphrases...")
        time.sleep(1)
        progress(0.3, desc="Paraphrases generated")
        time.sleep(0.5)
        
        # Phase 2: Getting responses
        progress(0.35, desc="Getting responses...")
        # Show incremental progress for responses
        for i in range(3):
            time.sleep(0.8)
            prog = 0.35 + (0.3 * ((i+1) / 3))
            progress(prog, desc=f"Getting responses ({i+1}/3)...")
        
        progress(0.65, desc="All responses received")
        time.sleep(0.5)
        
        # Phase 3: Analyzing
        progress(0.7, desc="Analyzing responses for hallucinations...")
        time.sleep(2)
        
        # Complete
        progress(1.0, desc="Analysis complete!")
        return "Process completed successfully!"
    
    with gr.Blocks() as demo:
        with gr.Row():
            btn = gr.Button("Start Process")
            output = gr.Textbox(label="Result")
        
        btn.click(fn=slow_process, outputs=output)
    
    demo.launch()

# Main application entry point
if __name__ == "__main__":
    logger.info("Starting PAS2 Hallucination Detector")
    interface = create_interface()
    logger.info("Launching Gradio interface...")
    interface.launch(
        show_api=False, 
        quiet=True,  # Changed to True for Hugging Face deployment
        share=False,
        max_threads=10,
        debug=False  # Changed to False for production deployment
    )
    
# Uncomment this line to run the test function instead of the main interface
# if __name__ == "__main__":
#     test_progress()