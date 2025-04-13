from pathlib import Path
import gradio as gr


def create_gradio_interface(answer_question, add_repository):
    css_file = Path("./gradio_style.css")

    with gr.Blocks(css=css_file.read_text()) as demo:
        with gr.Tabs():
            with gr.Tab("Задать вопрос", id="question_tab"):
                with gr.Row():
                    with gr.Column():
                        question_input = gr.Textbox(lines=2, placeholder="Задай вопрос по проекту...", label="Вопрос")
                        with gr.Accordion("Настройки модели", open=False):
                            temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                            max_tokens = gr.Slider(128, 4096, value=1024, step=128, label="Max Tokens")
                            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.1, label="Top-p")
                        with gr.Accordion("Настройки поиска", open=False):
                            k_results = gr.Slider(1, 10, value=5, step=1, label="Количество результатов")
                            fetch_k = gr.Slider(5, 50, value=20, step=5, label="Fetch K (кол-во кандидатов)")
                            mmr_lambda = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="MMR Lambda (разнообразие)")
                        with gr.Row():
                            ask_button = gr.Button("Спросить", variant="primary")
                            clear_btn = gr.Button("Очистить", variant="secondary")
                    with gr.Column():
                        answer_output = gr.Textbox(label="Ответ", interactive=False)

            with gr.Tab("Добавить репозиторий", id="repo_tab"):
                with gr.Row():
                    with gr.Column():
                        repo_url_input = gr.Textbox(
                            placeholder="https://github.com/username/repository.git", label="URL GitHub репозитория"
                        )
                        with gr.Accordion("Настройки чанкинга", open=False):
                            chunk_size = gr.Slider(128, 1024, value=512, step=64, label="Размер чанка (токены)")
                            chunk_overlap = gr.Slider(0, 256, value=50, step=16, label="Перекрытие чанков (токены)")
                        with gr.Row():
                            add_repo_button = gr.Button("Добавить репозиторий", variant="primary")
                            clear_repo_btn = gr.Button("Очистить", variant="secondary")
                    with gr.Column():
                        repo_status_output = gr.Textbox(label="Статус", interactive=False)

            with gr.Tab("Информация", id="info_tab"):
                gr.Markdown("### Инструкция по использованию")
                gr.Markdown(
                    """
                1. **Задать вопрос** - задавайте вопросы о вашем проекте
                2. **Добавить репозиторий** - загрузите новый репозиторий для анализа
                3. Настройте параметры модели и поиска для лучших результатов
                """
                )
                gr.Markdown("### Технологии")
                gr.Markdown(
                    """
                - **LM Studio** - локальный LLM
                - **LangChain** - обработка и анализ текста
                - **Qdrant** - векторное хранилище
                - **bge-m3** - модели эмбедингов
                """
                )

        # Обработчики кнопок очистки
        clear_btn.click(
            lambda: [None, 0.7, 1024, 0.9, 5, 20, 0.5],
            outputs=[question_input, temperature, max_tokens, top_p, k_results, fetch_k, mmr_lambda],
        )
        clear_repo_btn.click(lambda: [None, 512, 50], outputs=[repo_url_input, chunk_size, chunk_overlap])

        ask_button.click(
            answer_question,
            inputs=[question_input, temperature, max_tokens, top_p, k_results, fetch_k, mmr_lambda],
            outputs=answer_output,
        )
        add_repo_button.click(
            add_repository, inputs=[repo_url_input, chunk_size, chunk_overlap], outputs=repo_status_output
        )

    return demo