import json
import logging
import time
import os
from typing import List, Dict
import argparse
import csv
from pas2 import PAS2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_hallucination_evaluator(json_file: str, num_samples: int = None):
    pas2 = PAS2()  # Initialize the PAS2 library
    
    data = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        data.append(sample)
                        if num_samples and len(data) >= num_samples:
                            break
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON on line {line_num}: {e}")
                        continue
    except Exception as e:
        logger.error(f"Error reading the JSON file: {e}")
        return

    if not data:
        logger.error("No data to process.")
        return

    total_samples = len(data)
    correct_detections = 0
    processed_samples = 0
    detailed_results = []

    for idx, sample in enumerate(data):
        sample_id = sample.get('ID')
        user_query = sample.get('user_query')
        true_label = sample.get('hallucination')  # 'yes' or 'no'

        if user_query is None or true_label is None or sample_id is None:
            logger.warning(f"Sample {idx + 1} is missing 'ID', 'user_query' or 'hallucination' fields.")
            continue

        try:
            # Use the PAS2 library to detect hallucination
            hallucinated, initial_response, all_questions, all_responses = pas2.detect_hallucination(
                user_query, n_paraphrases=5, similarity_threshold=0.9, match_percentage_threshold=0.7
            )

            # Convert 'yes'/'no' to boolean for comparison
            true_hallucinated = true_label.strip().lower() == 'yes'

            # Compare the detected hallucination with the true label
            if hallucinated == true_hallucinated:
                correct_detections += 1

            processed_samples += 1

            # Store detailed result
            detailed_results.append({
                'ID': sample_id,
                'user_query': user_query,
                'true_label': true_label.lower(),
                'detected_hallucination': 'yes' if hallucinated else 'no',
                'initial_response': initial_response,
                'paraphrased_questions': all_questions[1:],  # Exclude the original question
                'all_responses': all_responses
            })

            # Optional: print progress every 10 samples
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{total_samples} samples...")

            # To avoid hitting rate limits (if applicable)
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error processing sample {idx + 1}: {e}")
            continue

    # Calculate accuracy
    accuracy = (correct_detections / processed_samples * 100) if processed_samples > 0 else 0
    logger.info(f"Processed Samples: {processed_samples}/{total_samples}")
    logger.info(f"Accuracy: {accuracy:.2f}%")

    # Save accuracy to a file
    with open('accuracy.txt', 'w', encoding='utf-8') as f:
        f.write(f"Processed Samples: {processed_samples}/{total_samples}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")

    # Save detailed results to a CSV file
    with open('detailed_results.csv', 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['ID', 'user_query', 'true_label', 'detected_hallucination', 'initial_response', 'paraphrased_questions', 'all_responses']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in detailed_results:
            writer.writerow({
                'ID': result['ID'],
                'user_query': result['user_query'],
                'true_label': result['true_label'],
                'detected_hallucination': result['detected_hallucination'],
                'initial_response': result['initial_response'],
                'paraphrased_questions': json.dumps(result['paraphrased_questions']),
                'all_responses': json.dumps(result['all_responses'])
            })

    logger.info("Evaluation complete. Results saved to 'accuracy.txt' and 'detailed_results.csv'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hallucination Evaluator using PAS2 library')
    parser.add_argument('--json_file', type=str, default='general_data.json', help='Path to the JSON file')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to process')
    args = parser.parse_args()

    run_hallucination_evaluator(args.json_file, args.num_samples)


# python pas2-benchmark.py --num_samples 10