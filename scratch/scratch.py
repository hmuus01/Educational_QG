from absl import flags

flags.DEFINE_string(
    "tpu_job_name", None,
    "Name of TPU worker binary. Only necessary if job name is changed from "
    "default tpu_worker.")
flags.DEFINE_string(
    "model_dir", "/tmp/transformer_standalone", "Estimator model_dir")


flags.DEFINE_string(
    "tpu", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

flags.DEFINE_string(
    "gcp_project",
    None,
    "Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_string(
    "tpu_zone", None,
    "GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_multi_string(
    "module_import", "t5.data.mixtures",
    "Modules to import. Use this, for example, to add new `Task`s to the "
    "global `TaskRegistry`.")

flags.DEFINE_string(
    "t5_tfds_data_dir", None,
    "If set, this directory will be used to store datasets prepared by "
    "TensorFlow Datasets that are not available in the public TFDS GCS bucket. "
    "Note that this flag overrides the `tfds_data_dir` attribute of all "
    "`Task`s.")

flags.DEFINE_list(
    "additional_task_cache_dirs", [],
    "Directories to search for Tasks in addition to defaults.")

flags.DEFINE_boolean("use_model_api", False,
                     "Use Model API instead of utils.run.")

flags.DEFINE_list("additional_deprecated_gin_references", [],
                  "Deprecated gin configs to be ignored.")

flags.DEFINE_boolean("skip_all_gin_unknowns", False,
                     "Don't throw any errors if any gin config params are "
                     "not found. Overrides the specific list of names in "
                     "--additional_deprecated_gin_references and "
                     "DEPRECATED_GIN_REFERENCES.")

# Note: All the args from here on are only used when use_model_api is set
flags.DEFINE_enum(
    "mode", None, ["train", "finetune", "eval", "predict",
                   "export_predict", "export_score", "score"],
    "Mode with which to run the model.")
flags.DEFINE_integer("batch_size", 1,
                     "Number of sequences per batch.")
flags.DEFINE_integer("input_sequence_length", 512,
                     "Number of tokens in input sequence.")
flags.DEFINE_integer("target_sequence_length", 512,
                     "Number of tokens in target sequence.")

# TPU-specific args.
flags.DEFINE_string("tpu_topology", "v2-8",
                    "The TPU topology being used. Ignored if --tpu not set.")
flags.DEFINE_integer("model_parallelism", 8,
                     "The number of cores per model replica. Ignored if --tpu "
                     "not set.")

# Train mode args
flags.DEFINE_integer("train_steps", 1000, "Number of training iterations.")
flags.DEFINE_string("mixture_or_task", "wmt_t2t_ende_v003",
                    "Name of Mixture or Task to use for training/evaluation.")
flags.DEFINE_string("pretrained_model_dir", "",
                    "Pretrained model dir for finetuning a model.")

# Eval mode args
flags.DEFINE_enum(
    "checkpoint_mode", "latest", ["all", "latest", "specific"],
    "Checkpoint steps to use when running 'eval', 'predict', 'finetune', and "
    "'export' modes. Can specify a list of checkpoints or all or the latest "
    "checkpoint. 'finetune' and 'export' modes work with 'latest' or "
    "'specific' with a single checkpoint.")
flags.DEFINE_list(
    "checkpoint_steps", [],
    "Checkpoint step numbers used for 'eval', 'predict', and 'finetune' modes. "
    "This argument is only used when which_checkpoint='specific'. "
    "For the 'finetune' mode, only a single checkpoint must be specified.")

flags.DEFINE_string("eval_summary_dir", "", "Path to save eval summaries")
flags.DEFINE_string("eval_split", "validation",
                    "Dataset split to use for evaluation.")

# Predict mode args
flags.DEFINE_string("input_file", "",
                    "Path to input file for decoding or scoring.")
flags.DEFINE_string("target_file", "", "Path to target file for scoring.")
flags.DEFINE_string("output_file", "", "Path to output file to save decodes.")

# Export mode args
flags.DEFINE_string(
    "export_dir", "",
    "Directory to export SavedModels to. Will use `model_dir` if unspecified.")


# Decoding strategy args, used in export and predict modes.
flags.DEFINE_integer("beam_size", 1, "Beam size for predict or export mode.")
flags.DEFINE_float("temperature", 0.0,
                   "Sampling emperature for predict or export mode.")
flags.DEFINE_integer("keep_top_k", -1,
                     "Top-k value for predict or export mode.")
