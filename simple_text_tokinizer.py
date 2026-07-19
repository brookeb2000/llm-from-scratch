# simple text tokinizer

import re

from matplotlib import text

# simple text tokenizer class
class SimpleTokenizerV2:
   def __init__(self, vocab):
       self.str_to_int = vocab
       self.int_to_str = { i:s for s,i in vocab.items()}
   
   def encode(self, text):
       preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
       preprocessed = [item.strip() for item in preprocessed if item.strip()]
       preprocessed = [item if item in self.str_to_int           
                       else "<|unk|>" for item in preprocessed]
       ids = [self.str_to_int[s] for s in preprocessed]
       return ids
       
   def decode(self, ids):
       text = " ".join([self.int_to_str[i] for i in ids])
       text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)           
       return text



# open file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

# split text into words and punctuation
import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

print(preprocessed[:30])

# create vocabulary
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))


tokenizer = SimpleTokenizerV2(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable 
pride."""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))

