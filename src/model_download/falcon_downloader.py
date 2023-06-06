from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def parse_version(version: str) -> str:
    """Converts user input from cli.py into Huggingface model names"""
    model_name = ''
    try:
        if version == '7b':
            model_name = "tiiuae/falcon-7b"
        elif version == '7b-inst':
            model_name = "tiiuae/falcon-7b-instruct"
        elif version == '40b':
            model_name = "tiiuae/falcon-40b"
        elif version == '40b-inst':
            model_name = "tiiuae/falcon-40b-instruct"
        else:
            raise ValueError(f"Sorry, but please provide a valid name. {version} is not valid. \n  "
                             f"Available are '7b', '7b-inst','40b', '40b-inst'")
    except ValueError as e:
        print(e)

    return model_name


def validate_directory(directory: str):
    try:
        if not os.path.isdir(directory):
            raise ValueError(f"Sorry, but please provide a valid directory path. \n "
                             f"{directory} does not seem to be a directory.")

    except ValueError as e:
        print(e)


def start_download(user_input_model_version: str, path: str):
    model_name = parse_version(user_input_model_version)

    validate_directory(path)

    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)



