import os
from git import Repo


def clone_repository(repo_url, target_dir="repo_clone"):
    try:
        os.makedirs(target_dir, exist_ok=True)

        print(f"Клонируем репозиторий {repo_url}...")
        Repo.clone_from(repo_url, target_dir)
        print("Клонирование завершено успешно!")

        return os.path.abspath(target_dir)

    except Exception as e:
        print(f"Ошибка при клонировании: {e}")
        return None
