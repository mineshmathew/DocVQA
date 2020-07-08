# Training / Testing BERT QA models on DocVQA
For finetuning pretrained BERT models, we use Simpletransformers which in turn is based on transformers

If you dont want to use Simpletransformers please follow instructions to finetune a QA model given here 

### Installing simpletransformers

For installing simpletransformers please follow steps here. You dont need nvidia apex unless you want to train with fp16
In our experiments we dont use fp16



### Data in SQuAD format
DocVQA being a VQA dataset is annotated with textual answers for each question. At the same time QA models using BERT require answers to be a span of the given context/paragraph/document.\
To this end we check if an answer is a substring of the serialized ocr transcription of the given document. And if yes, the index where the answer lies is taken as the start index for the answer. 
For answers not longer than 2 characters, we check for subsequence match instead of substring matching to avoid noisy substring matches when answers are smaller.

The data prepared in the SQuAD format, following the above steps are available in data_in_squad_format folder

### fine tuning a pretrained BERT model


### make predictions

pretrained model available here https://iiitaphyd-my.sharepoint.com/:u:/g/personal/minesh_mathew_research_iiit_ac_in/ERcV6gGX1OVBgy2ohwnRLLoBKCefBkhP_6CWfYiasVuOKQ?e=gkWQlO

