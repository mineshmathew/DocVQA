from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# training parameters; given below is the default setting used for bert large models
#for BERT large you need at least 4 GPUS with 11 GB of GPU momory
# for bert base models one GPU is sufficient for a batch size of 32
# make lr higher if you train with larger batch size

model_args = {"train_batch_size": 8, "n_gpu":4, "eval_batch_size": 64, 'max_answer_length': 50,  'num_train_epochs': 6, 'output_dir': "./output/", 'best_model_dir': './output/best_model/', 'evaluate_during_training': True, 'fp16': False, 'use_cached_eval_features':True, 'save_eval_checkpoints': False, 'save_model_every_epoch': False, 'max_seq_length': 384, 'doc_stride': 128, 'do_lower_case': True, 'gradient_accumulation_steps': 1, 'learning_rate': 2e-05 }


# if you want to fine tune a model locally saved or say you want to continue training a model previously saved give location of the dir where the model is
#model = QuestionAnsweringModel('bert', './models/bert-large-squad-docvqa-finetuned/', args=model_args)


# if you want to fine tune a pretrained model from pytorch trasnformers model zoo (https://huggingface.co/transformers/pretrained_models.html), you can directly give the model name ..the pretrained model will be downloadef first to a cache dir 
# here the model we are fine tuning is bert-large-cased-whole-word-masking-finetuned-squad
model = QuestionAnsweringModel('bert', 'bert-large-cased-whole-word-masking-finetuned-squad', args=model_args)

print (model.args)
#import pdb; pdb.set_trace()

# load train and val data
with open('./data_in_squad_format/docvqa_train_squad_format.json') as f:
  train_data = json.load(f)

with open('./data_in_squad_format/docvqa_val_squad_format.json') as f:
  dev_data = json.load(f)

#train 
model.train_model(train_data, show_running_loss= False, eval_data=dev_data)
~                                  
