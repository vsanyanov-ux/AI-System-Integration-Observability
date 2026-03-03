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

if __name__ == "__main__":
    system = SentimentAnalyzer()
    test_text = "This product is absolutely amazing and I love it!"
    prediction = system.analyze(test_text)
    print(f"Текст: {test_text}")
    print(f"Результат системы: {prediction}")
