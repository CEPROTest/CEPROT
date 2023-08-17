from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer.from_file(
    "./salesforce/codet5-vocab.json",
    "./salesforce/codet5-merges.txt"
)
tokenizer.add_special_tokens([
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<mask>"
])
