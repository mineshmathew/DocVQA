from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# predictions for each sample will be written to the "output_dir" specified below
# see "predictions_test.json" in the specified output_dir for per sample predictions
model_args = { "eval_batch_size": 256,  'output_dir': "./output/",  'doc_stride': 128, 'do_lower_case': True, 'max_seq_length': 384  }

#below give the folder where your trained model is
model = QuestionAnsweringModel('bert', './models/bert-large-squad-docvqa-finetuned/', args=model_args)

print (model.args)
#import pdb; pdb.set_trace()


with open('./data_in_squad_format/docvqa_test_squad_format.json') as f:
  json_data = json.load(f)

model.eval_model(json_data)

