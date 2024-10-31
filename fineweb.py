import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)


DATA_CACHE_DIR = os.path.join(os.getcwd(), local_dir)
os.makedirs(DATA_CACHE_DIR , exist_ok=True)

#download the dataset

#fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
# Define the specific parquet files you want to load (first seven)
data_files = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(3)]

# Load only these specific files from the dataset
fw = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    data_files=data_files,
    split="train"
)


#init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  #end of text token

def tokenize(doc):
  tokens = [eot]
  tokens.extend(enc.encode_ordinary(doc["text"]))
  tokens_np = np.array(tokens)
  assert  (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "tokens dictionary too big"
  tokens_np_uint16 = tokens_np.astype(np.uint16)
  return tokens_np_uint16

def write_datafile(filename, tokens_np):  # here we convert the file in binary number
  with open(filename, "wb") as f:
    f.write(tokens_np.tobytes())

nprocessor = max(1, int(os.cpu_count()/2))  #half of processor  available in your computer
with mp.Pool(nprocessor) as pool:
  shard_index = 0
  all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  #this is the array we created to keep the tokens as shard_size ie. 100 million tokens and save it 
  #here it is empty
  token_count = 0
  #progress bar is for animation of the downloading of the file or uploading the token in shard
  progress_bar = None
  for tokens in pool.imap(tokenize, fw, chunksize=8):  #here the func tokenize will act on data file fw and that tokenize data will one by one will go into the for loop for storing perpose it acts as generator 
    if token_count + len(tokens) < shard_size:  #this logic if the tokens size is below shard size token_count is for counting the token position or real time toekn count
      all_tokens_np[token_count:token_count + len(tokens)] = tokens
      token_count += len(tokens)  #progress the token count
      if progress_bar is None:  #we are adding the size to the progress bar to shard size
        progress_bar = tqdm(total=shard_size, unit="token", desc=f"Shard {shard_index}")
      progress_bar.update(len(tokens))
    #if the tokens is more that shard_size then it is handle here 
    else:
      split = "val" if shard_index == 0 else "train"
      filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")  
      remaining_token = shard_size - token_count
      progress_bar.update(remaining_token)
      all_tokens_np[token_count:token_count+remaining_token] = tokens[:remaining_token]
      write_datafile(filename, all_tokens_np)
      shard_index += 1
      progress_bar = None
      all_tokens_np[0:len(tokens)- remaining_token] = tokens[remaining_token:]
      token_count = len(tokens) - remaining_token

if token_count != 0:
  split = "val" if shard_index == 0 else "train"
  filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")
  write_datafile(filename, all_tokens_np[:token_count])
  if progress_bar:
                progress_bar.close()