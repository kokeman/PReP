# PReP: Pseudo-References filtered by Paraphrasing
Metric for Automatic Machine Translation Evaluation  
We submited it to [WMT2019 Metrics Shared Task](http://www.statmt.org/wmt19/metrics-task.html).  
Paper: http://www.statmt.org/wmt19/pdf/53/WMT60.pdf


# Dependencies
* Python >= 3.6.0
* TensorFlow >= 1.11.0.

# Downloads
   - Clone BERT repository (https://github.com/google-research/bert) and
   ```
   export PYTHONPATH="path to bert dir:$PYTHONPATH"
   ```

   - Download the [BERT model fine-tuned with MRPC](https://drive.google.com/file/d/1jMzKdbH2rz653DBjPU7Oy1dW_Dmdx61m/view?usp=sharing) and

   ```
   export TUNED_MODEL_DIR="path to fine-tuned BERT model"
   ```


# How to use
1. Prepare test set to ```data/orig```  (File names are src, out, ref)
2. Make pseudo-references  
  Translate the source of test set with off-the-shelf MT system and set the outputs to  ```data/pseudo_references/```

   Note: Don't use a off-the-shelf MT system whose output is contained the test set.   

3. Filtering with BERT  

   ```
   sh script/filter.sh
   ```

   Pseudo-references with paraphrase score are in ```data/sim_scores```.  
   Filtered pseudo-references are in ```data/filtered_paseudo_references/ ```.

4. Evaluate  
  Please evaluate the score with a metric which allows use of multiple references.  
  If you evaluate with sentence bleu, please download [moses binaries](http://www.statmt.org/moses/RELEASE-4.0/binaries/) and 

   ```
   sh scripts/evaluate.sh [language] [path to moses folder]
   ```
   Generated output_score file contains each sentence-bleu score.

   Note: If you evaluate the score with other metrics, use the metrics that take into account all references, not to get the maximum value for each single reference.
