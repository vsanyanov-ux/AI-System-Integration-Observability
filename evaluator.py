import mlflow
import pandas as pd
from app import SentimentAnalyzer
from sklearn.metrics import accuracy_score, classification_report
import json

def run_evaluation():
    # 1. Данные для оценки (Benchmark / Gold Standard)
    # ИИ-инженер фокусируется на качестве данных для оценки системы
    test_data = [
        {"text": "I love this!", "label": "POSITIVE"},
        {"text": "This is terrible.", "label": "NEGATIVE"},
        {"text": "I am so happy with the result.", "label": "POSITIVE"},
        {"text": "What a disaster.", "label": "NEGATIVE"},
        {"text": "The service was okay, not great.", "label": "NEGATIVE"}, # Сложный случай
    ]
    df = pd.DataFrame(test_data)

    # 2. Инициализация системы (Готового модуля)
    system = SentimentAnalyzer()

    # 3. Настройка MLflow (Observability)
    mlflow.set_experiment("AI_System_Integration_Evals")

    with mlflow.start_run():
        # Логируем версию системы как параметр
        mlflow.log_param("model_name", "distilbert-sst-2")
        mlflow.log_param("task", "sentiment-analysis")

        # 4. Прогон оценки
        predictions = []
        for text in df['text']:
            res = system.analyze(text)
            predictions.append(res['label'])

        # 5. Сбор метрик качества
        accuracy = accuracy_score(df['label'], predictions)
        report = classification_report(df['label'], predictions, output_dict=True)

        print(f"Evaluation Accuracy: {accuracy}")

        # 6. Логирование результатов
        mlflow.log_metric("eval_accuracy", accuracy)
        
        with open("eval_report.json", "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact("eval_report.json")

        print("Оценка завершена. Результаты в MLflow.")

if __name__ == "__main__":
    run_evaluation()
