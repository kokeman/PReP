TARGET_LANG=$1
MOSES_ROOT=$2
TOKENIZER=$MOSES_ROOT/scripts/tokenizer/tokenizer.perl
SENT_BLEU=$MOSES_ROOT/bin/sentence-bleu

mkdir -p data/tokenized/orig data/tokenized/filtered_pseudo_references

# tokenized
cat data/orig/ref | $TOKENIZER -l $TARGET_LANG -threads 5 | perl -pe '$_= lc($_)' > data/tokenized/orig/ref
cat data/orig/out | $TOKENIZER -l $TARGET_LANG -threads 5 | perl -pe '$_= lc($_)' > data/tokenized/orig/out
for file in $(ls data/filtered_pseudo_references/); do
    cat data/filtered_pseudo_references/$file | $TOKENIZER -l $TARGET_LANG -threads 5 | perl -pe '$_= lc($_)' > data/tokenized/filtered_pseudo_references/$file
done

# evaluate
$SENT_BLEU data/tokenized/orig/ref data/tokenized/filtered_pseudo_references/* < data/tokenized/orig/out > output_score

