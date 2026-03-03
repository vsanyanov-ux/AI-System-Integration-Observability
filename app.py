from transformers import pipeline

class SentimentAnalyzer:
    """
    Класс-обертка для готового модуля (Hugging Face Pipeline).
    ИИ-инженер не обучает эту модель, а встраивает её в систему.
    """
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        print(f"Загрузка готового модуля: {model_name}")
        self.analyzer = pipeline("sentiment-analysis", model=model_name)

    def analyze(self, text: str):
        result = self.analyzer(text)[0]
        return result

import mlflow

if __name__ == "__main__":
    system = SentimentAnalyzer()
    
    # Настройка Observability
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("AI_System_Interactive_Sessions")
    
    print("\n" + "="*50)
    print("🤖 AI Sentiment Analyzer (Connected to MLflow)")
    print("Type your English phrase to analyze.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye! 👋")
                break
                
            if not user_input:
                continue
                
            prediction = system.analyze(user_input)
            
            # Логируем каждый запрос в MLflow
            with mlflow.start_run():
                mlflow.log_param("user_input", user_input)
                mlflow.log_param("prediction_label", prediction['label'])
                mlflow.log_metric("confidence_score", prediction['score'])
                
            print(f"AI:  [{prediction['label']}] (Confidence: {prediction['score']:.1%})\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
