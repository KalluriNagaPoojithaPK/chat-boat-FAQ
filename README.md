import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data
nltk.download('punkt')

# FAQ Database
faqs = {
    "What is your return policy?": 
        "We accept returns within 30 days of purchase with original receipt.",
    "How do I track my order?": 
        "Track using the tracking number in your shipping confirmation email.",
    "What payment methods do you accept?": 
        "We accept Visa, Mastercard, PayPal, and Apple Pay.",
    "Do you offer international shipping?": 
        "Yes, we ship to over 50 countries worldwide.",
    "How can I contact customer service?": 
        "Call 1-800-555-1234 or email support@example.com."
}

def preprocess_text(text):
    """Clean and tokenize text"""
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

def get_best_answer(user_query):
    """Find matching FAQ using TF-IDF and cosine similarity"""
    questions = list(faqs.keys())
    processed_questions = [preprocess_text(q) for q in questions]
    processed_query = preprocess_text(user_query)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_questions + [processed_query])
    
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_idx = np.argmax(cosine_similarities)
    
    confidence = float(cosine_similarities[0][best_match_idx])
    return questions[best_match_idx], faqs[questions[best_match_idx]], confidence

def chatbot():
    print("\nFAQ Chatbot (type 'quit' to exit)")
    print("="*40)
    print("Available questions:")
    for i, q in enumerate(faqs.keys(), 1):
        print(f"{i}. {q}")
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
            
        matched_q, answer, confidence = get_best_answer(query)
        
        print(f"\nMatched Question: {matched_q}")
        print(f"Answer: {answer}")
        print(f"Confidence: {confidence * 100:.1f}%")

if __name__ == '__main__':
    chatbot()# chat-boat-FAQ
