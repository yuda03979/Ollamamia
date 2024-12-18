import ollama
from ollama import ListResponse
from tqdm import tqdm


def model(model_name, _from=None, _to=None, model_file_content=None) -> bool:
    local = True if _from else False  # right now support only pull
    if model_name not in ls(details=False, log=False):
        if not local:
            if pull(model_name):
                return True
        else:
            if add(model_name, _from, model_file_content):
                return True


def overwrite_to(_to):
    # move all the old models to the new dir
    pass


def ls(details=True, log=False) -> list:
    response: ListResponse = ollama.list()
    models_info = []
    for model_object in response.models:
        if details:
            models_info.append(model_object.model)
            continue
        temp = {'Name': model_object.model}
        if model_object.details and details:
            temp['Size (MB)'] = f'{(model_object.size.real / 1024 / 1024):.2f}'
            temp['Size vram (MB)'] = f'{(model_object.model.size_vram / 1024 / 1024):.2f}'
            temp['Digest'] = model_object.digest
            temp['Expires at'] = model_object.expires_at
            temp['Format'] = model_object.details.format
            temp['Family'] = model_object.details.family
            temp['Parameter Size'] = model_object.details.parameter_size
            temp['Quantization Level'] = model_object.details.quantization_level

        models_info.append(temp)
    if log:
        print([f"info\n" for info in models_info], "\n")
    return models_info


def pull(model_name) -> bool:
    current_digest, bars = '', {}
    for progress in ollama.pull(model_name, stream=True):
        digest = progress.get('digest', '')
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()

        if not digest:
            print(progress.get('status'))
            continue

        if digest not in bars and (total := progress.get('total')):
            bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

        if completed := progress.get('completed'):
            bars[digest].update(completed - bars[digest].n)

    current_digest = digest
    return True


def add(model_name, _from, modelfile=None) -> bool:
    modelfile = f"""
    FROM {_from}
    """
    example_modelfile = f"""
    FROM {model_name}
    PARAMETER temperature 0.1
    PARAMETER num_ctx 1536
    SYSTEM You are helpful assistant.
    """

    if not modelfile:
        modelfile = example_modelfile

    for response in ollama.create(model=model_name, modelfile=modelfile, stream=True):
        print(response['status'])
    return True
