import json
import torch
from configuration_chatglm import ChatGLMConfig
from tokenization_chatglm import ChatGLMTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration


if __name__ == '__main__':
    with open('./config.json') as f:
            conf = json.load(f)
    # Initializing a ChatGLM-6B THUDM/ChatGLM-6B style configuration
    configuration = ChatGLMConfig(**conf)
    tokenizer = ChatGLMTokenizer('./ice_text.model', num_image_tokens=0)
    # Initializing a model from the THUDM/ChatGLM-6B style configuration
    print(configuration.params_dtype)
    model = ChatGLMForConditionalGeneration(configuration)
    model.to('cpu')

    prompt = '你好'
    response, history = model.chat(tokenizer, prompt, history=[])
    print(response)
