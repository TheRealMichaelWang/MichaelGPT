import torch
import tiktoken
from datasets import load_dataset
from typing import Optional
from model import Config
import os

# basic dataset that feeds `context_window` number of tokens into LLM to predict a single token
# entire dataset is loaded into memory which can get a bit absurd
class LLMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        corpus: torch.Tensor,
        context_window: int,
        stride: int = 32,
        drop_last: bool = True,
    ):
        """
        corpus: concatenated token ids (1D tensor)
        context_window: sequence length for each sample
        stride: step (in tokens) between the starts of consecutive samples
        drop_last: if False, include a final sample aligned to the last possible start
        """
        if context_window <= 0:
            raise ValueError("context_window must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.corpus = corpus
        self.context_window = context_window
        self.stride = stride

        # last valid start index so that we have (x=context_window, y=context_window) tokens
        max_start = len(corpus) - (context_window + 1)
        if max_start < 0:
            raise ValueError(
                f"Corpus too short ({len(corpus)} tokens) for context_window={context_window}"
            )

        # precompute sample starts with the given stride
        self.starts = list(range(0, max_start + 1, stride))

        # optionally include a final window aligned to the very end
        if not drop_last and (not self.starts or self.starts[-1] != max_start):
            self.starts.append(max_start)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.context_window
        x = self.corpus[start:end]
        y = self.corpus[start + 1 : end + 1]
        return x, y
    
def load_llm_dataset(path: str, name: Optional[str], split: str, split2: str, config: Config) -> LLMDataset:
    ds = load_dataset(path, name, streaming=False)

    dataset_name = name if name is not None else "default"
    local_file = f"llm_dataset_{path.replace('/', '_')}_{dataset_name}_{split}.pt"
    if os.path.exists(local_file):
        corpus = torch.load(local_file, weights_only=False)
    else:
        ds = load_dataset(path, name, streaming=False)
        encoder = tiktoken.get_encoding("gpt2")

        tokens = []
        empty_streak = 0
        for example in ds[split][split2]:
            if example == "":
                empty_streak += 1
                
                if empty_streak >= 2:
                    tokens.append(config.pad_token_id)
                    empty_streak = 0
            else:
                if empty_streak > 0:
                    tokens.extend(encoder.encode('\n'))

                empty_streak = 0
                tokens.extend(encoder.encode(example))

        corpus = torch.tensor(tokens, dtype=torch.long)
        torch.save(corpus, local_file)
    return LLMDataset(corpus, config.context_window)

def load_wikitext103(config: Config):
    return load_llm_dataset("wikitext", "wikitext-103-raw-v1", "train", "text", config), load_llm_dataset("wikitext", "wikitext-103-raw-v1", "validation", "text", config)

def load_wikitext2(config: Config):
    return load_llm_dataset("wikitext", "wikitext-2-raw-v1", "train", "text", config), load_llm_dataset("wikitext", "wikitext-2-raw-v1", "validation", "text", config)