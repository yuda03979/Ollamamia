from ollama import Client


class Globals:
    host = 'http://localhost:11434'
    len_logs = 100
    available_tasks = ["null", "generate", "chat", "stream_chat", "embed"]

    def __init__(self):
        self.client = None

    def init(self):
        self.client = Client(host=self.host)


GLOBALS = Globals()
