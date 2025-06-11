from datasets import load_dataset
from tqdm import tqdm
import tiktoken

encoder = tiktoken.encoding_for_model('gpt2')

ds = load_dataset("roneneldan/TinyStories")
print(ds)
print(ds['train'][0])

save_path = "train_data2.txt"

tokens = encoder.encode("""Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. Beep was a healthy car because he always had good fuel. Good fuel made Beep happy and strong.\n', '\n', 'One day, Beep was driving in the park when he saw a big tree. The tree had many leaves that were falling. Beep liked how the leaves fall and wanted to play with them. Beep drove under the tree and watched the leaves fall on him. He laughed and beeped his horn.\n', '\n', 'Beep played with the falling leaves all day. When it was time to go home, Beep knew he needed more fuel. He went to the fuel place and got more healthy fuel. Now, Beep was ready to go fast and play again the next day. And Beep lived happily ever after.\n""")
print(len(tokens))
# with open(save_path, 'r') as f:
#     s = f.readlines()
    
# print(s[0:10])

# # string = ""
# for data in tqdm(ds['train'], "PROCESSING"):
#     string = f"{data['text']}\n"
#     with open(save_path, "a") as f:
#         f.write(string) 

# print(f"Saved as {save_path}")
# # print(ds['valid'])
# # print(ds['test'])

