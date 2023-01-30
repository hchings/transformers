from transformers import RobertaConfig

# config = RobertaConfig.from_pretrained("roberta-base", vocab_size=50265)
config = RobertaConfig.from_pretrained("klue/roberta-small", vocab_size=50265)


config.save_pretrained("./roberta-base")
