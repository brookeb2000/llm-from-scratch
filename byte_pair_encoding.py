from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

#create tokenizer
tokenizer = tiktoken.get_encoding("gpt2")


# example converting text to tokens and back
text = "Akwirw ier"

integers = tokenizer.encode(text)
print(integers)

strings = tokenizer.decode(integers)

print(strings)

# example encoding a file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4 #A
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")



import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
   def __init__(self, txt, tokenizer, max_length, stride):
       self.input_ids = []
       self.target_ids = []
       token_ids = tokenizer.encode(txt)                         #tokenize the entire text
       for i in range(0, len(token_ids) - max_length, stride):   # Use a sliding window to chunk the book into overlapping sequences of max_length
           input_chunk = token_ids[i:i + max_length]
           target_chunk = token_ids[i + 1: i + max_length + 1]
           self.input_ids.append(torch.tensor(input_chunk))
           self.target_ids.append(torch.tensor(target_chunk))
   
   def __len__(self):                                            # Return the total number of rows in the dataset
       return len(self.input_ids)
   
   def __getitem__(self, idx):                                   # Return a single row from the dataset
       return self.input_ids[idx], self.target_ids[idx]
   
   
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
   
import torch
print(torch.__version__)