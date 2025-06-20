from zhipuai import ZhipuAI
import time
import re
from typing import List, Dict


class GLM4_API:
    def __init__(self, api_key, batch_size, semantic_prompt, expression_prompt,vocabulary_prompt):
        self.batch_size = batch_size
        self.client = ZhipuAI(api_key=api_key)
        self.initial_score = 0.01
        self.model = 'glm-4-plus'
        self.max_retries = 40
        self.semantic_prompt = semantic_prompt
        self.expression_prompt = expression_prompt
        self.vocabulary_prompt = vocabulary_prompt

    def evaluate_semantic_similarity(self, batch, responses):
        task_ids = []
        results = []
        for ori, para in zip(batch['batch_original_sentence'], responses):
            combined_prompt = ori + para + self.semantic_prompt
            response = self.client.chat.asyncCompletions.create(
                model=self.model,
                messages=[{"role": "user", "content": combined_prompt}]
            )
            task_ids.append(response.id)
        
        results = self.polling_task(task_ids, self.max_retries)
        return results
    
    def evaluate_expression_style(self, responses):
        task_ids = []
        results = []    
        for ori in responses:
            combined_prompt = ori + self.expression_prompt
            response = self.client.chat.asyncCompletions.create(
                model=self.model,
                messages=[{"role": "user", "content": combined_prompt}]
            )
            task_ids.append(response.id)
        
        results = self.polling_task(task_ids, self.max_retries)
        return results

    def vocabulary_filtering(self, responses):
        task_ids = []
        results = []    
        for ori in responses:
            combined_prompt = ori + self.vocabulary_prompt
            response = self.client.chat.asyncCompletions.create(
                model=self.model,
                messages=[{"role": "user", "content": combined_prompt}]
            )
            task_ids.append(response.id)

        results = self._poll_tasks(task_ids)
        return results

    def polling_task(self, task_ids, max_retries):
        results = [self.initial_score] * self.batch_size 
        for _ in range(max_retries):
            all_done = True
            for idx, task_id in enumerate(task_ids):
                if results[idx] != self.initial_score:  
                    continue
                try:
                    resp = self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
                    if resp.task_status == "SUCCESS":
                        content = resp.choices[0].message.content
                        match = re.search(r"(-?0\.\d)|-?1(.0)?|0", content)
                        if match:
                            score = float(match.group())
                            results[idx] = max(-1.0, min(1.0, score))
                        else:
                            results[idx] = 0.0
                    elif resp.task_status == "FAILED":
                        results[idx] = 0.0
                    else:
                        all_done = False
                except Exception as e:
                    results[idx] = 0.0
            
            if all_done:
                break
            time.sleep(0.5)
        return results
    
    def _poll_tasks(self, task_ids: List[str]) -> List[str]:
        results = [""] * len(task_ids)
        for _ in range(self.max_retries):
            all_done = True
            
            for idx, task_id in enumerate(task_ids):
                if results[idx]:  
                    continue
                try:
                    resp = self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
                    if resp.task_status == "SUCCESS":
                        results[idx] = resp.choices[0].message.content.strip()
                    elif resp.task_status == "FAILED":
                        results[idx] = f"Error processing {task_id}"
                    else:
                        all_done = False
                except Exception as e:
                    results[idx] = f"API Error: {str(e)}"
            if all_done:
                break
            time.sleep(0.5)
        return results
        


