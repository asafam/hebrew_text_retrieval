from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import argparse

# Preparing the dataset
def load_hf_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset

# Use only 'entailment' pairs (label==2) for MultipleNegativesRankingLoss
def make_examples(
        dataset, 
        include_negative=False,
        entailment = "entailment",
        contradiction = "contradiction",
        premise = "translation1",
        hypothesis = "translation2",
        label = "original_label"
    ):
    examples = [
        InputExample(texts=[item[premise], item[hypothesis]], label=1.0)
        for item in dataset
        if item[label] == entailment and item[premise] and item[hypothesis]
    ]
    if include_negative:
        examples += [
            InputExample(texts=[item[hypothesis], item[premise]], label=0.0)
            for item in dataset
            if item[label] == contradiction and item[premise] and item[hypothesis]
        ]
    return examples

def main(
        dataset_name,
        model_name_or_path,
        max_seq_length = 128,
        batch_size = 512,
        epochs = 1,
        evaluation_steps = 100,
        output_path = './outputs/models/sbert/sbert-modernbert-nli',
        save_best_model = True,
        show_progress_bar = True,
        print_loss_callback = None,
        train_split = 'train',
        validation_split = 'validation'
):  
    # Load the dataset
    dataset = load_hf_dataset(dataset_name)
    train_examples = make_examples(dataset[train_split])
    val_examples   = make_examples(dataset[validation_split], include_negative=True)

    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    train_dataloader = DataLoader(train_examples, shuffle=True,  batch_size=batch_size)

    # Model setup
    bert = models.Transformer(
        model_name_or_path=model_name_or_path, 
        tokenizer_name_or_path=model_name_or_path,
        max_seq_length=max_seq_length,
    )
    pooling = models.Pooling(
        bert.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False
    )
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
        evaluation_steps=evaluation_steps,
        epochs=epochs,
        output_path=output_path,
        save_best_model=save_best_model,
        show_progress_bar=show_progress_bar,
        callback=print_loss_callback,
    )

if __name__ == "__main__":
    argparse.ArgumentParser(description="Train an SBERT model on Hebrew NLI dataset using Multiple Negatives Ranking Loss.")
    parser = argparse.ArgumentParser(description="Train an SBERT model on Hebrew NLI dataset using Multiple Negatives Ranking Loss.")
    parser.add_argument("--dataset_name", type=str, default="HebArabNlpProject/HebNLI", help="Name of the dataset to use")
    parser.add_argument("--model_name_or_path", type=str, default="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250522_1841", help="Pretrained model name or path")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--evaluation_steps", type=int, default=100, help="Number of steps between evaluations")
    parser.add_argument("--train_split", type=str, default='train', help="Name of the training split in the dataset")
    parser.add_argument("--validation_split", type=str, default='dev', help="Name of the validation split in the dataset")
    parser.add_argument("--output_path", type=str, help="Path to save the trained model")
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        evaluation_steps=args.evaluation_steps,
        train_split=args.train_split,
        validation_split=args.validation_split,
        output_path=args.output_path
    )
