
def generate_template(template, add_token_num):
    prompt_token = template
    assert add_token_num >= 2
    for i in range(3, add_token_num + 1):
        token = '<template_{}>'
        prompt_token += token.format(str(i))
    # print(prompt_token)
    return prompt_token

def extend_tokenizers(tokenizer, add_token_num):
    prompt_tokens = []
    for i in range(1, add_token_num + 1):
        prompt_tokens.append('<template_{}>'.format(i))

    tokenizer.add_special_tokens({"additional_special_tokens": prompt_tokens})
    return tokenizer