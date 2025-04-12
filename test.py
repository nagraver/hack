from openai import OpenAI
import os

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
project_path = "./repo_clone"

def analyze_single_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code_content = f.read()

        response = client.chat.completions.create(
            model="llama-3.1-8b-lexi-uncensored-v2",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You're a senior code analyst. Analyze the provided code and describe:\n"
                        "1. Основное назначение файла\n"
                        "2. Ключевые функции/классы\n"
                        "3. Входные/выходные данные (если применимо)\n"
                        "4. Зависимости (если видны)\n"
                        "5. Особенности реализации\n"
                        "Будь лаконичным (3-5 предложений).\n"
                        "Отвечай на русском языке."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Проанализируй этот код:\n\n{code_content}",
                },
            ],
            temperature=0.2,
        )

        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        return "Не удалось проанализировать код"

    except Exception as e:
        return f"Ошибка при анализе файла {file_path}: {str(e)}"

def analyze_code_project(repo_path: str) -> str:
    allowed_extensions = (
        ".py", ".md", ".txt", ".js", ".java", 
        ".cpp", ".h", ".html", ".css",
    )
    
    analysis_results = []
    total_files = 0
    analyzed_files = 0
    
    for root, dirs, files in os.walk(repo_path):
        if ".git" in dirs:
            dirs.remove(".git")

        for file in files:
            if not file.lower().endswith(allowed_extensions):
                continue
            total_files += 1
            file_path = os.path.join(root, file)
            
            # Анализируем каждый файл отдельно
            file_analysis = analyze_single_file(file_path)
            analysis_results.append(f"Файл: {file_path}\nАнализ: {file_analysis}\n")
            analyzed_files += 1
    
    if not analysis_results:
        return "В проекте не найдено файлов для анализа."
    
    # Создаем суммарный отчет
    summary = (
        f"Общий анализ проекта: {repo_path}\n"
        f"Всего файлов: {total_files}\n"
        f"Проанализировано файлов: {analyzed_files}\n\n"
        "Детальный анализ по файлам:\n\n" + 
        "\n".join(analysis_results)
    )
    
    return summary

if __name__ == "__main__":
    analysis_result = analyze_code_project(project_path)
    print("\nРезультат анализа проекта:")
    print(analysis_result)