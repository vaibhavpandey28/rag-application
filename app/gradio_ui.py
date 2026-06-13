import json
import requests
import gradio as gr

API_URL = "http://127.0.0.1:8000/api/v1/chat/completion"


def chat(message, history):

    answer = ""

    try:
        with requests.post(
            API_URL,
            json={"query": message},
            stream=True,
            timeout=300,
            headers={
                "Accept": "text/event-stream",
            },
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):

                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                try:
                    payload = json.loads(line[6:])
                except Exception:
                    continue

                if payload.get("done"):
                    break

                answer += payload.get("content", "")

                yield answer

    except Exception as e:
        yield f"❌ {str(e)}"


demo = gr.ChatInterface(
    fn=chat,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(
        placeholder="Message Liza...",
        container=True,
    ),
    title="💜 Liza AI",
    description="FastAPI + LangGraph Chat",
)

if __name__ == "__main__":
    demo.launch()