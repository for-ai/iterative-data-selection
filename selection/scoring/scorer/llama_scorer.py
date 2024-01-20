
from .base import Scorer

class Llama_Scorer(Scorer):
    
    @property
    def id2score(self):
        
        id2score = {
                29896: "1",
                29906: "2",
                29941: "3",
                29946: "4",
                29945: "5",
                29953: "6"
                }
        
        return id2score
    
    @property
    def complexity_template(self):
        
        complexity_template = ("You are a helpful assistant. Please identify the complexity score of the following user query. \n##Query: {instruction}  \n##Complexity: ")
        
        return complexity_template
    
    @property
    def quality_template(self):
        
        quality_template = ("You are a helpful assistant. Please identify the quality score of the Response corresponding to the Question. \n #Question#:\n{instruction}\n#Response#:\n{output} \n##Quality: ")
        
        return quality_template