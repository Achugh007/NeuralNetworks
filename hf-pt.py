#Working code - Final version

import logging
from transformers import pipeline, logging

logging.set_verbosity_error()

# Sentiment Analysis (PyTorch)
sentiment_classifier = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment') 
# ... (rest of the code)

# Question Answering (PyTorch)
context = "Hugging Face is headquartered in New York City."
question = "Where is Hugging Face based?"
qa_pipeline = pipeline('question-answering', model='deepset/minilm-uncased-squad2')  
answer = qa_pipeline(question=question, context=context)
print(answer) 

