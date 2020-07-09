\[readme work in progress\]
# Training / Testing BERT QA models on DocVQA
For finetuning pretrained BERT models, we use [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers) which in turn is based on [transformers](https://github.com/huggingface/transformers) package.

If you dont want to use Simpletransformers please follow instructions to finetune a QA model given [here](https://github.com/huggingface/transformers/tree/master/examples/question-answering) 

### Installing simpletransformers

For installing simpletransformers please follow [original instructions](https://github.com/ThilinaRajapakse/simpletransformers#setup)
We dont use fp16 while finetuning BERT models in our experiments. You need not install Apex if you dont want to use fp16 training.




### Data in SQuAD format
DocVQA is annotated in the same way as reading comprehension datasets like [SQuAD](https://arxiv.org/abs/1606.05250)   are annotated. SQuAD for example
treats reading comprehension as an extractive QA problem where ground truth answers are  marked as a 'span' of the context paragraph.\

So to benchmark BERT QA model on DocVQA, we approximate the annotated answers in DocVQA train split as span of the OCR transcription of the given document image.

To this end we check if an answer is a substring of the serialized ocr transcription ( OCR transcription of all images in the dataset, using a commercial OCR is provides as part of the dataset)  of the given document. And if yes, the index where the answer lies is taken as the start index for the answer. This is akin to what Python's ``` find() ``` function does.
For answers not longer than 2 characters, we check for subsequence match instead of substring matching to avoid noisy substring matches when answers are smaller. For example if answer is "1" it will get matched with  a "1" in "1998" or a "1" in "1up".
Using the above answer span approximation technique we could find answer as a span of the OCR transcription for 32526 questions out of 39463 questions in original train split of DocVQA. For val and test splits too, a SQuAD style json is prepared so that the data can be easily used with existing reading comprehension codes.  The data prepared in the SQuAD format, following the above steps are available in ```data_in_squad_format``` folder

### Fine tune a pretrained BERT model on DocVQA 
See the script ```finetune_pretrained_model.py```

### Make predictions using an existing BERT QA model
See the script ```test_model.py```
If you want to make predictions using a QA model already finetuned on DocVQA. Please download a model which is a  ```bert-large-uncased-whole-word-masking-finetuned-squad```   from [transformers pretrained moels zoo](https://huggingface.co/transformers/pretrained_models.html), finetuned DocVQA train split. This model, based on our experiments yield 0.665 ANLS on the test split of DocVQA.\
Download the model from [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/minesh_mathew_research_iiit_ac_in/ERcV6gGX1OVBgy2ohwnRLLoBKCefBkhP_6CWfYiasVuOKQ?e=gkWQlO)

