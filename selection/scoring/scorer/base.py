import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class Scorer(object):
    
    def __init__(self, model_name_or_path: str, is_vllm: bool  = False):
        
        self.is_vllm = is_vllm
        
        if not is_vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        else:
            
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(model_name_or_path, disable_log_stats = True, tensor_parallel_size=2)
            self.sampling_params = SamplingParams(max_tokens = 512, logprobs = 1000)
        
    def infer_score(self, user_input: str):

        max_length = 512
        
        if self.is_vllm:
            outputs = self.llm.generate(user_input, self.sampling_params, use_tqdm = False)
            score_template = np.array([1,2,3,4,5,6])
            
            try:
                logprobs_list = outputs[0].outputs[0].logprobs[0]
            except IndexError:
                logger.warning("Meeting Index Error. Returning A Placeholder Score -1.")
                return -1
        else:
            input_ids = self.tokenizer.encode(user_input, return_tensors = "pt")
            outputs = self.model.generate(input_ids, max_length = max_length, num_return_sequences = 1, return_dict_in_generate = True, output_scores = True)
            logprobs_list = outputs.scores[0][0]
            
        score_logits = []
        score_template = np.array([1,2,3,4,5,6])
        for k in self.id2score:
            score_logits.append(logprobs_list[k])
        score_logits = np.array(score_logits)
        score_npy = softmax(score_logits, axis=0)
        score_npy = score_npy * score_template

        score_npy = np.sum(score_npy, axis=0)
        
        return score_npy
    
    def infer_batch_score(self, user_input_list: list):
            
            max_length = 512
            
            if self.is_vllm:
                outputs = self.llm.generate(user_input_list, self.sampling_params, use_tqdm = False) # output is a list of Meeting objects
                score_template = np.array([1,2,3,4,5,6])
                
                batch_logprobs_list = []
                for output in outputs:
                    try:
                        logprobs_list = output.outputs[0].logprobs[0]
                        batch_logprobs_list.append(logprobs_list)
                    except IndexError:
                        logger.warning("Meeting Index Error. Returning A Placeholder Score -1.")
                        batch_logprobs_list.append(-1)
            else:
                input_ids = self.tokenizer.batch_encode_plus(user_input_list, return_tensors = "pt", padding = True, truncation = True, max_length = max_length)
                outputs = self.model.generate(input_ids["input_ids"], max_length = max_length, num_return_sequences = 1, return_dict_in_generate = True, output_scores = True)
                logprobs_list = outputs.scores[0][0]
                
            scores = []
            for logprobs in batch_logprobs_list:
                if logprobs == -1:
                    scores.append(-1)
                    continue
                score_logits = []
                score_template = np.array([1,2,3,4,5,6])
                for k in self.id2score:
                    score_logits.append(logprobs[k])
                score_logits = np.array(score_logits)
                score_npy = softmax(score_logits, axis=0)
                score_npy = score_npy * score_template
        
                score_npy = np.sum(score_npy, axis=0)
                scores.append(score_npy)

            return scores
    
    def infer_complexity(self, input_text: str):
        
        complexity_template = self.complexity_template
        user_input = complexity_template.format(instruction=input_text)
        
        return self.infer_score(user_input)
        
    def infer_quality(self, input_text: str, resp_text: str):
        
        quality_template = self.quality_template
        user_input = quality_template.format(instruction=input_text, output=resp_text)
        
        return self.infer_score(user_input)

    def batch_infer(self, input_text_list: list, resp_text_list: list):
        """
        input_text_list: list of input text
        resp_text_list: list of response text
        """
        quality_template = self.quality_template
        complexity_template = self.complexity_template
        quality_list = []
        complexity_list = []
        for input_text, resp_text in zip(input_text_list, resp_text_list):
            user_input = quality_template.format(instruction=input_text, output=resp_text)
            quality_list.append(user_input)
            user_input = complexity_template.format(instruction=input_text)
            complexity_list.append(user_input)

        quality_scores = self.infer_batch_score(quality_list)
        complexity_scores = self.infer_batch_score(complexity_list)
        return quality_scores, complexity_scores
    
    @property
    def id2score(self):
        raise NotImplementedError
    
    @property
    def complexity_template(self):
        raise NotImplementedError
    
    @property
    def quality_template(self):
        raise NotImplementedError
    
# 