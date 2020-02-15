# Herbert - Polish RoBERTA model


[Named after great polish poet Zbigniew Herbert](https://en.wikipedia.org/wiki/Zbigniew_Herbert)


Polish RoBERTA model trained on Polish Wikipedia, Polish literature, Oscar.


## Training Fairseq Polish RoBERTA from scratch



1. Prepare huge text file 'data.txt' with Wikipedia text, each wiki article is separated by new line
2.  Prepare another file for sentencpiece, where each sentence is in one line. 
3. Train sentencepiece vocabulary. 
Do you use default mapping for custom symbols Unknown (<unk>), BOS (<s>) and EOS </s>) or add others eg. <pad> or <mask> https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md

4. Parse vocabulary and save it in fairseq format vocab.fairseq.txt
5. Encode 'data.txt' with trained sentencepiece model data.sp.txt
6. preprocsse data.sp.txt with fairseq-preproces
```
fairseq-preprocess \
    --only-source \
    --srcdict vocab.fairseq.txt \
    --trainpref data.sp.txt \
     --destdir data-bin/wikitext \
    --workers 60
```

Do I have to provide 
```
 --trainpref wikitext-103-raw/wiki.train.bpe \
 --validpref wikitext-103-raw/wiki.valid.bpe \
 --testpref wikitext-103-raw/wiki.test.bpe \
```
or trainpref is enough

7. Run training 
```
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x

DATA_DIR=data-bin/wikitext

fairseq-train --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1