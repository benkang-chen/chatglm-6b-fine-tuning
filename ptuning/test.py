from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

if __name__ == '__main__':
    # Load dataset
    # data_files = {}
    # train_file = 'AdvertiseGen/train.json'
    # validation_file = 'AdvertiseGen/dev.json'
    # if train_file is not None:
    #     data_files["train"] = train_file
    #     extension = train_file.split(".")[-1]
    # if validation_file is not None:
    #     data_files["validation"] = validation_file
    #     extension = validation_file.split(".")[-1]
    # print(data_files)
    # print(extension)
    # raw_datasets = load_dataset(
    #     extension,
    #     cache_dir=None,
    #     data_files=data_files,
    #     use_auth_token=None
    # )
    # print(raw_datasets)
    # column_names = raw_datasets["validation"].column_names
    # print(column_names)
    # train_dataset = raw_datasets["train"]
    # print(len(train_dataset))
    # train_dataset = train_dataset.map(
    #     lambda x: {"a": x['content']},
    #     batched=True,
    #     num_proc=1,
    #     remove_columns=column_names,
    #     load_from_cache_file=False,
    #     desc="Running tokenizer on train dataset",
    # )
    # print(train_dataset)
    # for data in train_dataset:
    #     print(data)
    model_name_or_path = '..\\chatglm-6b\\'
    # model_name_or_path = 'THUDM/chatglm-6b'
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir='./test', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir='./test', trust_remote_code=True)
    print(tokenizer('你好'))
