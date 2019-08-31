import run_classifier
import tokenization
import tensorflow as tf
import modeling
import argparse
import os


def load_file(path):
    list_ = []
    for line in open(path):
        list_.append(line.rstrip())
    return list_


def load_files(path):
    files_dict = {}  # key: file name, value: return of load_file()
    for file_name in os.listdir(path):
        if os.path.isfile(path + '/' + file_name):
            files_dict[file_name] = load_file(path + '/' + file_name)
    return files_dict


def get_predict_examples(list1, list2):
    examples = []
    for i, (sent1, sent2) in enumerate(zip(list1, list2)):
        guid = f"test-{i}"
        text_a = tokenization.convert_to_unicode(sent1)
        text_b = tokenization.convert_to_unicode(sent2)
        label = "0"
        examples.append(run_classifier.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def mrpc_classifier(sent_list1, sent_list2, args):
    TUNED_MODEL_DIR = args.tuned_model_dir

    config = {
        "task_name": 'MRPC',
        "do_predict": True,
        "vocab_file": f"{TUNED_MODEL_DIR}/vocab.txt",
        "bert_config_file": f"{TUNED_MODEL_DIR}/bert_config.json",
        "init_checkpoint": f"{TUNED_MODEL_DIR}",
        "max_seq_length": 128,
        "output_dir": f"{TUNED_MODEL_DIR}",
        "do_lower_case": True,
        "predict_batch_size": 8
    }

    bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
    processor = run_classifier.MrpcProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=config["vocab_file"], do_lower_case=config["do_lower_case"])

    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=config["output_dir"],
        save_checkpoints_steps=1000,
    )

    model_fn = run_classifier.model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=config["init_checkpoint"],
        learning_rate=5e-5,
        num_train_steps=None,
        num_warmup_steps=None,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=config["predict_batch_size"]
    )

    predict_examples = get_predict_examples(sent_list1, sent_list2)
    num_actual_predict_examples = len(predict_examples)

    predict_file = os.path.join(config["output_dir"], "predict.tf_record")
    run_classifier.file_based_convert_examples_to_features(
        predict_examples, label_list, config["max_seq_length"], tokenizer, predict_file
    )

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", config["predict_batch_size"])

    predict_drop_remainder = False
    predict_input_fn = run_classifier.file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=config["max_seq_length"],
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    probabilities = [prediction["probabilities"][1] for (i, prediction) in enumerate(result)]
    return probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_ref")
    parser.add_argument("pseudo_ref_dir")
    parser.add_argument("output_dir")
    parser.add_argument("tuned_model_dir")
    args = parser.parse_args()

    gold_ref = load_file(args.gold_ref)
    ref_dict = load_files(args.pseudo_ref_dir)

    for sys_name, ref in ref_dict.items():
        probs = mrpc_classifier(ref, gold_ref, args)

        path = f'{args.output_dir}/{sys_name}'
        with open(path, "w") as f:
            for prob, ref in zip(probs, ref):
                f.write(f"{ref}\t{prob}\n")


if __name__ == '__main__':
    main()
