
# for the docker:
# start ollama-serve
# fast api

model_name = "gemma2:2b"
from src.ollamamia import Ollamamia
ollamamia = Ollamamia(on_docker=False)
model_config = ollamamia.model_config(model_name=model_name, task="generate")
model_config.options.temperature = 0.0
ollamamia[model_name] = model_config

for i in range(2):
    print(ollamamia[model_name] << "hello  how are you today? id love if you will tell me story.")
print("finish!")
