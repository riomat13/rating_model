# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = '3.0.2'

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging

# Configurations
from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, AutoConfig
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from .configuration_utils import PretrainedConfig

# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_apex_available,
    is_py3nvml_available,
)


# Model Cards
from .modelcard import ModelCard

# TF 2.0 <=> PyTorch conversion utilities
from .modeling_tf_pytorch_utils import (
    convert_tf_weight_name_to_pt_weight_name,
    load_pytorch_checkpoint_in_tf2_model,
    load_pytorch_model_in_tf2_model,
    load_pytorch_weights_in_tf2_model,
    load_tf2_checkpoint_in_pytorch_model,
    load_tf2_model_in_pytorch_model,
    load_tf2_weights_in_pytorch_model,
)

# Pipelines
from .pipelines import (
    CsvPipelineDataFormat,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    JsonPipelineDataFormat,
    NerPipeline,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    pipeline,
)

# Tokenizers
from .tokenization_albert import AlbertTokenizer
from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from .tokenization_bert import BasicTokenizer, BertTokenizer, BertTokenizerFast, WordpieceTokenizer
from .tokenization_reformer import ReformerTokenizer
from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer, TransfoXLTokenizerFast
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TensorType,
    TokenSpan,
)
from .tokenization_utils_fast import PreTrainedTokenizerFast

# Trainer
from .trainer_utils import EvalPrediction, set_seed
from .training_args import TFTrainingArguments


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TensorFlow
from .generation_tf_utils import tf_top_k_top_p_filtering
from .modeling_tf_utils import (
    shape_list,
    TFPreTrainedModel,
    TFSequenceSummary,
    TFSharedEmbeddings,
)
from .modeling_auto import (
    TF_MODEL_MAPPING,
    TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    TF_MODEL_FOR_PRETRAINING_MAPPING,
    TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    TF_MODEL_WITH_LM_HEAD_MAPPING,
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_MASKED_LM_MAPPING,
    TFAutoModel,
    TFAutoModelForMultipleChoice,
    TFAutoModelForPreTraining,
    TFAutoModelForQuestionAnswering,
    TFAutoModelForSequenceClassification,
    TFAutoModelForTokenClassification,
    TFAutoModelWithLMHead,
    TFAutoModelForCausalLM,
    TFAutoModelForMaskedLM,
)

from .modeling_albert import (
    TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    TFAlbertForMaskedLM,
    TFAlbertForMultipleChoice,
    TFAlbertForPreTraining,
    TFAlbertForQuestionAnswering,
    TFAlbertForSequenceClassification,
    TFAlbertForTokenClassification,
    TFAlbertMainLayer,
    TFAlbertModel,
    TFAlbertPreTrainedModel,
)

from .modeling_bert import (
    TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    TFBertEmbeddings,
    TFBertLMHeadModel,
    TFBertForMaskedLM,
    TFBertForMultipleChoice,
    TFBertForNextSentencePrediction,
    TFBertForPreTraining,
    TFBertForQuestionAnswering,
    TFBertForSequenceClassification,
    TFBertForTokenClassification,
    TFBertMainLayer,
    TFBertModel,
    TFBertPreTrainedModel,
)

from .modeling_transfo_xl import (
    TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
    TFAdaptiveEmbedding,
    TFTransfoXLLMHeadModel,
    TFTransfoXLMainLayer,
    TFTransfoXLModel,
    TFTransfoXLPreTrainedModel,
)

# Optimization
from .optimization import (
    AdamWeightDecay,
    create_optimizer,
    GradientAccumulator,
    WarmUp,
)

# Trainer
from .trainer import TFTrainer

# Benchmarks
from .benchmark.benchmark import TensorFlowBenchmark
from .benchmark.benchmark_args import TensorFlowBenchmarkArguments
