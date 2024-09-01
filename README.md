# LLM finetuning for nostalgic youtube comments


```markdown
# YouTube Comments Sentiment Analysis

This project implements a sentiment analysis model to classify YouTube comments as either "nostalgia" or "not nostalgia" using Hugging Face's Transformers library. The model is fine-tuned on a custom dataset of YouTube comments.

## Overview of the Code

The code begins by installing the necessary libraries, including `datasets`, `transformers`, `peft`, and `evaluate`. These libraries provide the tools needed for data handling, model training, and evaluation.

```python
! pip install datasets
! pip install transformers
! pip install peft
! pip install evaluate
```

Next, the dataset is loaded using the `datasets` library, which retrieves a collection of YouTube comments for sentiment analysis. The dataset is then converted into a Pandas DataFrame for easier manipulation.

```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset('Senem/Nostalgic_Sentiment_Analysis_of_YouTube_Comments_Data')
df = pd.DataFrame(dataset['train'])
```

The sentiment labels in the DataFrame are mapped to numerical values, where "nostalgia" is represented by 1 and "not nostalgia" by 0. The comments are stored in a new column called `text`, and the labels are stored in a `label` column.

```python
df['sentiment'] = df['sentiment'].map({'nostalgia': 1, 'not nostalgia': 0})
df['text'] = df['comment']
df['label'] = df['sentiment']
df.drop(['comment', 'sentiment'], axis=1, inplace=True)
```

### Tokenization

The `tokenizer` from the Hugging Face library is initialized, and a function is defined to tokenize the text data. This function generates `input_ids` and `attention_mask` for each comment, which are essential for model training.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(text):
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True)
```

The tokenized data is then added to the DataFrame, separating `input_ids` and `attention_mask` into their respective columns.

```python
df[['input_ids', 'attention_mask']] = df['text'].apply(lambda x: pd.Series(tokenize_function(x)))
```

### Model Training

A DistilBERT model is initialized for sequence classification. The model is trained using the Hugging Face `Trainer` API, which simplifies the training process.

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/models/FineTuned/',
    num_train_epochs=15,
    per_device_train_batch_size=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
```

### Predictions

After training, the model is used to make predictions on new comments. The comments are tokenized, and the model outputs logits, which are converted to sentiment labels.

```python
text_list = ["I remember how we used to dance to his songs", "This is not good."]
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    print(f"{text} - {id2label[predictions.item()]}")
```

### Conclusion

This project demonstrates how to build a sentiment analysis model using Hugging Face's Transformers library, from data loading and preprocessing to model training and prediction. The model can be further fine-tuned and evaluated based on specific requirements.
```

