
import time

import gradio
import numpy as np
import torch
from transformers import LogitsProcessor

from modules import html_generator, shared
import time
import traceback

import torch
from transformers import is_torch_npu_available, is_torch_xpu_available

from modules import models, sampler_hijack, shared
from modules.logging_colors import logger
from modules.models import load_model
from modules.text_generation import generate_reply

params = {
    'active': True,
    'verbose': False  # For debugging mostly
}

class PerplexityLogitsData:
    def __init__(self):
        self.generated_token_ids = []
        self.selected_probs = []
        self.top_token_ids_list = []
        self.top_probs_list = []
        self.perplexities_list = []
        self.last_probs = None
        
        
    def calculate_perplexity(self, input_ids, scores):
        probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        log_probs = torch.nan_to_num(torch.log(probs))  # Note: This is to convert log(0) nan to 0, but probs*log_probs makes this 0 not affect the perplexity.
        entropy = -torch.sum(probs * log_probs)
        entropy = entropy.cpu().numpy()
        perplexity = round(float(np.exp(entropy)), 4)
        self.perplexities_list.append(perplexity)
        last_token_id = int(input_ids[0][-1].cpu().numpy().item())
        # Store the generated tokens (not sure why this isn't accessible in the output endpoint!)
        self.generated_token_ids.append(last_token_id)
        # Get last probability, and add to the list if it wasn't there
        if len(self.selected_probs) > 0:
            # Is the selected token in the top tokens?
            if self.verbose:
                print('Probs: Token after', shared.tokenizer.decode(last_token_id))
                print('Probs:', [shared.tokenizer.decode(token_id) for token_id in self.top_token_ids_list[-1][0]])
                print('Probs:', [round(float(prob), 4) for prob in self.top_probs_list[-1][0]])
            if last_token_id in self.top_token_ids_list[-1][0]:
                idx = self.top_token_ids_list[-1][0].index(last_token_id)
                self.selected_probs.append(self.top_probs_list[-1][0][idx])
            else:
                self.top_token_ids_list[-1][0].append(last_token_id)
                last_prob = round(float(self.last_probs[last_token_id]), 4)
                self.top_probs_list[-1][0].append(last_prob)
                self.selected_probs.append(last_prob)
        else:
            self.selected_probs.append(1.0)  # Placeholder for the last token of the prompt

        if self.verbose:
            pplbar = "-"
            if not np.isnan(perplexity):
                pplbar = "*" * round(perplexity)
            print(f"PPL: Token after {shared.tokenizer.decode(last_token_id)}\t{perplexity:.2f}\t{pplbar}")

        # Get top 5 probabilities
        top_tokens_and_probs = torch.topk(probs, 5)
        top_probs = top_tokens_and_probs.values.cpu().numpy().astype(float).tolist()
        top_token_ids = top_tokens_and_probs.indices.cpu().numpy().astype(int).tolist()

        self.top_token_ids_list.append(top_token_ids)
        self.top_probs_list.append(top_probs)

        probs = probs.cpu().numpy().flatten()
        self.last_probs = probs  # Need to keep this as a reference for top probs

class PerplexityLogits(LogitsProcessor):
    def __init__(self, verbose=False):
        self.raw = PerplexityLogitsData()
        self.sampled = PerplexityLogitsData()
        self.verbose = verbose

    def __call__(self, input_ids, scores):
        # t0 = time.time()
        
        # sampled 
        is_non_hf_exllamav2 = shared.model.__class__.__name__ == 'Exllamav2Model'
        is_non_hf_llamacpp = shared.model.__class__.__name__ == 'LlamaCppModel'
        if not any([is_non_hf_exllamav2, is_non_hf_llamacpp]):
            sampled_scores = sampler_hijack.global_scores[-1]        
            self.sampled.calculate_perplexity(input_ids, sampled_scores)

        # raw
        
        raw_scores = scores
        self.raw.calculate_perplexity(input_ids, raw_scores)
        # TODO: figure out if shared.model contains the model's logits pre-sampling
        
        # t1 = time.time()
        # print(f"PPL Processor: {(t1-t0):.3f} s")
        # About 1 ms, though occasionally up to around 100 ms, not sure why...
        # Doesn't actually modify the logits!
        return scores


# Stores the perplexity and top probabilities
ppl_logits_processor = None


def logits_processor_modifier(logits_processor_list, input_ids):
    global ppl_logits_processor
    if params['active']:
        ppl_logits_processor = PerplexityLogits(verbose=params['verbose'])
        logits_processor_list.append(ppl_logits_processor)

def ui():
    def update_active_check(x):
        params.update({'active': x})
        
    def update_verbose_check(x):
        params.update({'verbose': x})


    active_check = gradio.Checkbox(value=True, label="Compute probabilities and perplexity scores", info="Activate this extension. Note that this extension currently does not work with exllama or llama.cpp.")
    verbose_check = gradio.Checkbox(value=True, label="Verbose", info="Prints more information about the perplexity and probabilities in the console.")

    active_check.change(update_active_check, active_check, None)
    verbose_check.change(update_verbose_check, verbose_check, None)