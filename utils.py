import sparknlp
from torch import nn, load
from transformers import BertModel, BertTokenizer, BertForTokenClassification
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
import re
import numpy as np
import pandas as pd


stop_words = open("stop_words.txt", 'r').read().split()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = "cpu"
classification_model = None
# spark = sparknlp.start()
# p_model = None


def clean_special(input_text):
    text = input_text.split(".")
    text_after_filter2 = set()
    for sent in text:
        data = sent.lower()
        data = re.sub(r"[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~]+\ *", " ", data)
        data = re.sub("\\s+", " ", data).strip().split()
        filtered_sentence = [w for w in data if w not in stop_words]
        if len(filtered_sentence) > 3:
            text_after_filter2.add(" ".join(filtered_sentence))
    return list(text_after_filter2)


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.sig = nn.Sigmoid()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.sig(linear_output)
        return final_layer


def do_classification(diagnosis_text):
    global classification_model

    if not classification_model:
        classification_model = BertClassifier()
        classification_model.load_state_dict(load("model_w.pt", map_location=device))
        classification_model.eval()

    texts = [tokenizer(text, padding='max_length', max_length=25, truncation=True, return_tensors="pt")
             for text in diagnosis_text]

    tt = np.zeros(2)
    for tet in texts:
        mask = tet['attention_mask'].to(device)
        input_id = tet['input_ids'].squeeze(1).to(device)
        output = classification_model(input_id, mask)
        tt += output[0].data.numpy()
    cate = tt.argmax()
    tt /= len(texts)

    return_str = None
    if tt[cate] - tt[1 - cate] < 0.2 or tt[cate] < 0.5:
        return_str = "Could not classify clearly. Please enter more information."
    elif cate == 0:
        tt = np.around(tt * 100, decimals=3)
        return_str = "The probability that you have gotten COPD is: " + str(tt[1 - cate]) + "%.\n"
        return_str += "You may NOT have COPD."
    else:
        tt = np.around(tt * 100, decimals=3)
        return_str = "The probability that you have gotten COPD is: " + str(tt[cate]) + "%.\n"
        return_str += "You may HAVE COPD."

    return return_str


def do_NER(diagnosis_text):
    global spark
    global p_model

    if not p_model:
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
        sentence_detector = SentenceDetectorDLModel().setInputCols("document").setOutputCol("sentence")
        tokenizer2 = Tokenizer().setInputCols("sentence").setOutputCol("token")
        token_classifier = BertForTokenClassification.load(f"./model_tf_spark_nlp") \
            .setInputCols("token", "sentence") \
            .setOutputCol("label") \
            .setCaseSensitive(True)
        ner_converter = NerConverter().setInputCols(["sentence", "token", "label"]).setOutputCol("ner_chunk")

        pipeline = Pipeline(
            stages=[document_assembler, sentence_detector, tokenizer2, token_classifier, ner_converter]
        )
        p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

    result = p_model.transform(spark.createDataFrame([[diagnosis_text]]).toDF('text'))
    result = result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
        .select(F.expr("cols['0']").alias("Possible Symptoms"))

    return result.toJSON().first()
