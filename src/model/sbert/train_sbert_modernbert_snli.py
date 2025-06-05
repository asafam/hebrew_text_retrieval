from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader


from datasets import load_dataset
from sentence_transformers import InputExample

# Preparing the dataset
snli = load_dataset('snli')

# Use only 'entailment' pairs (label==2) for MultipleNegativesRankingLoss
def make_examples(dataset, include_negative=False):
    examples = [
        InputExample(texts=[item['premise'], item['hypothesis']], label=1.0)
        for item in dataset
        if item['label'] == 0 and item['premise'] and item['hypothesis']
    ]
    if include_negative:
        examples += [
            InputExample(texts=[item['hypothesis'], item['premise']], label=0.0)
            for item in dataset
            if item['label'] == 2 and item['premise'] and item['hypothesis']
        ]
    return examples

train_examples = make_examples(snli['train'])
val_examples   = make_examples(snli['validation'], include_negative=True)

print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

train_dataloader = DataLoader(train_examples, shuffle=True,  batch_size=512)

# Model setup
model_name = "answerdotai/ModernBERT-base"
bert = models.Transformer(
    model_name, 
    max_seq_length=8192,
)
pooling = models.Pooling(bert.get_word_embedding_dimension())
sbert_model = SentenceTransformer(modules=[bert, pooling])

# Loss function
train_loss = losses.MultipleNegativesRankingLoss(sbert_model)

# Validation
sents1 = [ex.texts[0] for ex in val_examples]
sents2 = [ex.texts[1] for ex in val_examples]
labels = [ex.label for ex in val_examples]

val_evaluator = EmbeddingSimilarityEvaluator(sents1, sents2, labels, name='validation')

def print_loss_callback(score, epoch, steps):
    print(f"Epoch: {epoch}, Step: {steps}, Validation Score: {score}")

# Train the model
sbert_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=val_evaluator,
    evaluation_steps=100,
    epochs=1,
    output_path='./outputs/models/sbert/sbert-modernbert-nli',
    save_best_model=True,
    show_progress_bar=True,
    callback=print_loss_callback,
)