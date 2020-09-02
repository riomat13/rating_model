# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
import json
import logging
import os
import pickle
import sys
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import chain
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from uuid import UUID

import numpy as np

from .configuration_auto import AutoConfig
from .configuration_utils import PretrainedConfig
from .modelcard import ModelCard
from .tokenization_auto import AutoTokenizer
from .tokenization_bert import BasicTokenizer
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import BatchEncoding, PaddingStrategy


import tensorflow as tf
from .modeling_auto import (
    TFAutoModel,
    TFAutoModelForSequenceClassification,
    TFAutoModelForQuestionAnswering,
    TFAutoModelForTokenClassification,
    TFAutoModelWithLMHead,
    TF_MODEL_WITH_LM_HEAD_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    TFAutoModelForCausalLM,
)

if TYPE_CHECKING:
    from .modeling_utils import PreTrainedModel
    from .modeling_tf_utils import TFPreTrainedModel


logger = logging.getLogger(__name__)


class PipelineException(Exception):
    """
    Raised by pipelines when handling __call__
    """

    def __init__(self, task: str, model: str, reason: str):
        super().__init__(reason)

        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling varargs for each Pipeline
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultArgumentHandler(ArgumentHandler):
    """
    Default varargs argument parser handling parameters for each Pipeline
    """

    @staticmethod
    def handle_kwargs(kwargs: Dict) -> List:
        if len(kwargs) == 1:
            output = list(kwargs.values())
        else:
            output = list(chain(kwargs.values()))

        return DefaultArgumentHandler.handle_args(output)

    @staticmethod
    def handle_args(args: Sequence[Any]) -> List[str]:

        # Only one argument, let's do case by case
        if len(args) == 1:
            if isinstance(args[0], str):
                return [args[0]]
            elif not isinstance(args[0], list):
                return list(args)
            else:
                return args[0]

        # Multiple arguments (x1, x2, ...)
        elif len(args) > 1:
            if all([isinstance(arg, str) for arg in args]):
                return list(args)

            # If not instance of list, then it should instance of iterable
            elif isinstance(args, Iterable):
                return list(chain.from_iterable(chain(args)))
            else:
                raise ValueError(
                    "Invalid input type {}. Pipeline supports Union[str, Iterable[str]]".format(type(args))
                )
        else:
            return []

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0 and len(args) > 0:
            raise ValueError("Pipeline cannot handle mixed args and kwargs")

        if len(kwargs) > 0:
            return DefaultArgumentHandler.handle_kwargs(kwargs)
        else:
            return DefaultArgumentHandler.handle_args(args)


class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing.
    Supported data formats currently includes:
     - JSON
     - CSV
     - stdin/stdout (pipe)

    PipelineDataFormat also includes some utilities to work with multi-columns like mapping from datasets columns
    to pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.
    """

    SUPPORTED_FORMATS = ["json", "csv", "pipe"]

    def __init__(
        self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False,
    ):
        self.output_path = output_path
        self.input_path = input_path
        self.column = column.split(",") if column is not None else [""]
        self.is_multi_columns = len(self.column) > 1

        if self.is_multi_columns:
            self.column = [tuple(c.split("=")) if "=" in c else (c, c) for c in self.column]

        if output_path is not None and not overwrite:
            if exists(abspath(self.output_path)):
                raise OSError("{} already exists on disk".format(self.output_path))

        if input_path is not None:
            if not exists(abspath(self.input_path)):
                raise OSError("{} doesnt exist on disk".format(self.input_path))

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: dict):
        """
        Save the provided data object with the representation for the current `DataFormat`.
        :param data: data to store
        :return:
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.
        :param data: data to store
        :return: (str) Path where the data has been saved
        """
        path, _ = os.path.splitext(self.output_path)
        binary_path = os.path.extsep.join((path, "pickle"))

        with open(binary_path, "wb+") as f_output:
            pickle.dump(data, f_output)

        return binary_path

    @staticmethod
    def from_str(
        format: str, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False,
    ):
        if format == "json":
            return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "csv":
            return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "pipe":
            return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        else:
            raise KeyError("Unknown reader {} (Available reader are json/csv/pipe)".format(format))


class CsvPipelineDataFormat(PipelineDataFormat):
    def __init__(
        self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

    def __iter__(self):
        with open(self.input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.is_multi_columns:
                    yield {k: row[c] for k, c in self.column}
                else:
                    yield row[self.column[0]]

    def save(self, data: List[dict]):
        with open(self.output_path, "w") as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)


class JsonPipelineDataFormat(PipelineDataFormat):
    def __init__(
        self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

        with open(input_path, "r") as f:
            self._entries = json.load(f)

    def __iter__(self):
        for entry in self._entries:
            if self.is_multi_columns:
                yield {k: entry[c] for k, c in self.column}
            else:
                yield entry[self.column[0]]

    def save(self, data: dict):
        with open(self.output_path, "w") as f:
            json.dump(data, f)


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process.
    For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}
    """

    def __iter__(self):
        for line in sys.stdin:
            # Split for multi-columns
            if "\t" in line:

                line = line.split("\t")
                if self.column:
                    # Dictionary to map arguments
                    yield {kwargs: l for (kwargs, _), l in zip(self.column, line)}
                else:
                    yield tuple(line)

            # No dictionary to map arguments
            else:
                yield line

    def save(self, data: dict):
        print(data)

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        if self.output_path is None:
            raise KeyError(
                "When using piped input on pipeline outputting large object requires an output file path. "
                "Please provide such output path through --output argument."
            )

        return super().save_binary(data)


class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()


class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations.
    Pipeline workflow is defined as a sequence of the following operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (Task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument. Users can specify
    device argument as an integer, -1 meaning "CPU", >= 0 referring the CUDA device ordinal.

    Some pipeline, like for instance FeatureExtractionPipeline ('feature-extraction') outputs large
    tensor object as nested-lists. In order to avoid dumping such large structure as textual data we
    provide the binary_output constructor argument. If set to True, the output will be stored in the
    pickle format.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
        binary_output (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e. pickle) or as raw text.

    Return:
        :obj:`List` or :obj:`Dict`:
        Pipeline returns list or dictionary depending on:

         - Whether the user supplied multiple samples
         - Whether the pipeline exposes multiple fields in the output object
    """

    default_input_names = None

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        task: str = "",
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
    ):

        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.modelcard = modelcard
        self.device = device
        self.binary_output = binary_output
        self._args_parser = args_parser or DefaultArgumentHandler()

        # Update config with task specific parameters
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None and task in task_specific_params:
            self.model.config.update(task_specific_params.get(task))

    def save_pretrained(self, save_directory):
        """
        Save the pipeline's model and tokenizer to the specified save_directory
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.
        example:
            # Explicitly ask for tensor allocation on CUDA device :0
            nlp = pipeline(..., device=0)
            with nlp.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = nlp(...)
        Returns:
            Context manager
        """
        with tf.device("/CPU:0" if self.device == -1 else "/device:GPU:{}".format(self.device)):
            yield

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.
        :param inputs:
        :return:
        """
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    def check_model_type(self, supported_models):
        """
        Check if the model class is in the supported class list of the pipeline.
        """
        if not isinstance(supported_models, list):  # Create from a model mapping
            supported_models = [item[1].__name__ for item in supported_models.items()]
        if self.model.__class__.__name__ not in supported_models:
            raise PipelineException(
                self.task,
                self.model.base_model_prefix,
                f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are {supported_models}",
            )

    def _parse_and_tokenize(self, *args, padding=True, add_special_tokens=True, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self._args_parser(*args, **kwargs)
        inputs = self.tokenizer(
            inputs, add_special_tokens=add_special_tokens, return_tensors='tf', padding=padding,
        )

        return inputs

    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        return self._forward(inputs)

    def _forward(self, inputs, return_tensors=False):
        """
        Internal framework specific forward dispatching.
        Args:
            inputs: dict holding all the keyworded arguments for required by the model forward method.
            return_tensors: Whether to return native framework (tf) tensors rather than numpy array.
        Returns:
            Numpy array
        """
        # Encode for forward
        with self.device_placement():
            # TODO trace model
            predictions = self.model(inputs.data, training=False)[0]

        if return_tensors:
            return predictions
        else:
            return predictions.numpy()


class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline using Model head. This pipeline extracts the hidden states from the base transformer,
    which can be used as features in downstream tasks.

    This feature extraction pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "feature-extraction", for extracting features of a sequence.

    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    `huggingface.co/models <https://huggingface.co/models>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        task: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
        )

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs).tolist()


class TextGenerationPipeline(Pipeline):
    """
    Language generation pipeline using any ModelWithLMHead head. This pipeline predicts the words that will follow a specified text prompt.

    This language generation pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "text-generation", for generating text from a specified prompt.

    The models that this pipeline can use are models that have been trained with an autoregressive language modeling objective,
    which includes the uni-directional models in the library (e.g. gpt2).
    See the list of available community models on
    `huggingface.co/models <https://huggingface.co/models?search=&filter=lm-head>`__.
    """

    # Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
    # in https://github.com/rusiaaman/XLNet-gen#methodology
    # and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e

    PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
    (except for Alexei and Maria) are discovered.
    The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
    remainder of the story. 1883 Western Siberia,
    a young Grigori Rasputin is asked by his father and a group of men to perform magic.
    Rasputin has a vision and denounces one of the men as a horse thief. Although his
    father initially slaps him for making such an accusation, Rasputin watches as the
    man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
    with people, even a bishop, begging for his blessing. """

    ALLOWED_MODELS = [
        "XLNetLMHeadModel",
        "TransfoXLLMHeadModel",
        "ReformerModelWithLMHead",
        "GPT2LMHeadModel",
        "OpenAIGPTLMHeadModel",
        "CTRLLMHeadModel",
        "TFXLNetLMHeadModel",
        "TFTransfoXLLMHeadModel",
        "TFGPT2LMHeadModel",
        "TFOpenAIGPTLMHeadModel",
        "TFCTRLLMHeadModel",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_model_type(self.ALLOWED_MODELS)

    # overriding _parse_and_tokenize to allow for unusual language-modeling tokenizer arguments

    def _parse_and_tokenize(self, *args, padding=True, add_special_tokens=True, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            tokenizer_kwargs = {"add_space_before_punct_symbol": True}
        else:
            tokenizer_kwargs = {}
        inputs = self._args_parser(*args, **kwargs)
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors="tf",
            padding=padding,
            **tokenizer_kwargs,
        )

        return inputs

    def __call__(
        self, *args, return_tensors=False, return_text=True, clean_up_tokenization_spaces=False, **generate_kwargs
    ):

        text_inputs = self._args_parser(*args)

        results = []
        for prompt_text in text_inputs:
            # Manage correct placement of the tensors
            with self.device_placement():
                if self.model.__class__.__name__ in [
                    "XLNetLMHeadModel",
                    "TransfoXLLMHeadModel",
                    "TFXLNetLMHeadModel",
                    "TFTransfoXLLMHeadModel",
                ]:
                    # For XLNet and TransformerXL we had an article to the prompt to give more state to the model.
                    padding_text = self.PADDING_TEXT + self.tokenizer.eos_token
                    padding = self._parse_and_tokenize(padding_text, padding=False, add_special_tokens=False)
                    # This impacts max_length and min_length argument that need adjusting.
                    padding_length = padding["input_ids"].shape[-1]
                    if "max_length" in generate_kwargs and generate_kwargs["max_length"] is not None:
                        generate_kwargs["max_length"] += padding_length
                    if "min_length" in generate_kwargs and generate_kwargs["min_length"] is not None:
                        generate_kwargs["min_length"] += padding_length

                    inputs = self._parse_and_tokenize(
                        padding_text + prompt_text, padding=False, add_special_tokens=False
                    )
                else:
                    inputs = self._parse_and_tokenize(prompt_text, padding=False, add_special_tokens=False)

                # set input_ids to None to allow empty prompt
                if inputs["input_ids"].shape[-1] == 0:
                    inputs["input_ids"] = None
                    inputs["attention_mask"] = None

                input_ids = inputs["input_ids"]

                # Ensure that batch size = 1 (batch generation not allowed for now)
                assert (
                    input_ids is None or input_ids.shape[0] == 1
                ), "Batch generation is currently not supported. See https://github.com/huggingface/transformers/issues/3021 for more information."

                output_sequences = self.model.generate(input_ids=input_ids, **generate_kwargs)  # BS x SL

            result = []
            for generated_sequence in output_sequences:
                if generated_sequence is not None:
                    generated_sequence = generated_sequence.cpu()
                generated_sequence = generated_sequence.numpy().tolist()
                record = {}
                if return_tensors:
                    record["generated_token_ids"] = generated_sequence
                if return_text:
                    # Decode text
                    text = self.tokenizer.decode(
                        generated_sequence,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )

                    # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                    if input_ids is None:
                        prompt_length = 0
                    else:
                        prompt_length = len(
                            self.tokenizer.decode(
                                input_ids[0],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                            )
                        )

                    record["generated_text"] = prompt_text + text[prompt_length:]

                result.append(record)
            results += [result]

        if len(results) == 1:
            return results[0]

        return results


class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using ModelForSequenceClassification head. See the
    `sequence classification usage <../usage.html#sequence-classification>`__ examples for more information.

    This text classification pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "sentiment-analysis", for classifying sequences according to positive or negative sentiments.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=text-classification>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __init__(self, return_all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.check_model_type(
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
        )

        self.return_all_scores = return_all_scores

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)
        if self.return_all_scores:
            return [
                [{"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(item)]
                for item in scores
            ]
        else:
            return [
                {"label": self.model.config.id2label[item.argmax()], "score": item.max().item()} for item in scores
            ]


class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",")]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError("You must include at least one label and at least one sequence.")
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                (
                    'The provided hypothesis_template "{}" was not able to be formatted with the target labels. '
                    "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
                ).format(hypothesis_template)
            )

        if isinstance(sequences, str):
            sequences = [sequences]
        labels = self._parse_labels(labels)

        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

        return sequence_pairs


class ZeroShotClassificationPipeline(Pipeline):
    """
    NLI-based zero-shot classification pipeline using a ModelForSequenceClassification head with models trained on
    NLI tasks.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pre-trained model. Then logit for `entailment` is then taken as the logit for the
    candidate label being valid. Any NLI model can be used as long as the first output logit corresponds to
    `contradiction` and the last to `entailment`.

    This pipeline can currently be loaded from the :func:`~transformers.pipeline` method using the following task
    identifier(s):

    - "zero-shot-classification"

    The models that this pipeline can use are models that have been fine-tuned on a Natural Language Inference task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?search=nli>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __init__(self, args_parser=ZeroShotClassificationArgumentHandler(), *args, **kwargs):
        super().__init__(*args, args_parser=args_parser, **kwargs)

    def _parse_and_tokenize(self, *args, padding=True, add_special_tokens=True, **kwargs):
        """
        Parse arguments and tokenize only_first so that hypothesis (label) is not truncated
        """
        inputs = self._args_parser(*args, **kwargs)
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors="tf",
            padding=padding,
            truncation="only_first",
        )

        return inputs

    def __call__(self, sequences, candidate_labels, hypothesis_template="This example is {}.", multi_class=False):
        """
        NLI-based zero-shot classification. Any combination of sequences and labels can be passed and each
        combination will be posed as a premise/hypothesis pair and passed to the pre-trained model. Then logit for
        `entailment` is then taken as the logit for the candidate label being valid. Any NLI model can be used as
        long as the first output logit corresponds to `contradiction` and the last to `entailment`.

        Args:
            sequences (:obj:`str` or obj:`List`):
                The sequence or sequences to classify. Truncated if model input is too large.
            candidate_labels (:obj:`str` or obj:`List`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (obj:`str`, defaults to "This example is {}."):
                The template used to turn each label into an NLI-style hypothesis. This template must include a {}
                or similar syntax for the candidate label to be inserted into the template. For example, the default
                template is "This example is {}." With the candidate label "sports", this would be fed into the model
                like `<cls> sequence to classify <sep> This example is sports . <sep>`. The default template works
                well in many cases, but it may be worthwhile to experiment with different templates depending on the
                task setting.
            multi_class (obj:`bool`, defaults to False):
                When False, it is assumed that only one candidate label can be true, and the scores are normalized
                such that the sum of the label likelihoods for each sequence is 1. When True, the labels are
                considered independent and probabilities are normalized for each candidate by doing a of softmax of
                the entailment score vs. the contradiction score.
        """
        outputs = super().__call__(sequences, candidate_labels, hypothesis_template)
        num_sequences = 1 if isinstance(sequences, str) else len(sequences)
        candidate_labels = self._args_parser._parse_labels(candidate_labels)
        reshaped_outputs = outputs.reshape((num_sequences, len(candidate_labels), -1))

        if len(candidate_labels) == 1:
            multi_class = True

        if not multi_class:
            # softmax the "entailment" logits over all candidate labels
            entail_logits = reshaped_outputs[..., -1]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)
        else:
            # softmax over the entailment vs. contradiction dim for each label independently
            entail_contr_logits = reshaped_outputs[..., [0, -1]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
            scores = scores[..., 1]

        result = []
        for iseq in range(num_sequences):
            top_inds = list(reversed(scores[iseq].argsort()))
            result.append(
                {
                    "sequence": sequences if isinstance(sequences, str) else sequences[iseq],
                    "labels": [candidate_labels[i] for i in top_inds],
                    "scores": scores[iseq][top_inds].tolist(),
                }
            )

        if len(result) == 1:
            return result[0]
        return result


class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using ModelWithLMHead head. See the
    `masked language modeling usage <../usage.html#masked-language-modeling>`__ examples for more information.

    This mask filling pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "fill-mask", for predicting masked tokens in a sequence.

    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=lm-head>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        topk=5,
        task: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
        )

        self.check_model_type(TF_MODEL_WITH_LM_HEAD_MAPPING)

        self.topk = topk

    def ensure_exactly_one_mask_token(self, masked_index: np.ndarray):
        numel = np.prod(masked_index.shape)
        if numel > 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"More than one mask_token ({self.tokenizer.mask_token}) is not supported",
            )
        elif numel < 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"No mask_token ({self.tokenizer.mask_token}) found on the input",
            )

    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        outputs = self._forward(inputs, return_tensors=True)

        results = []
        batch_size = outputs.shape[0]

        for i in range(batch_size):
            input_ids = inputs["input_ids"][i]
            result = []

            masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()

            # Fill mask pipeline supports only one ${mask_token} per sample
            self.ensure_exactly_one_mask_token(masked_index)

            logits = outputs[i, masked_index.item(), :]
            probs = tf.nn.softmax(logits)
            topk = tf.math.top_k(probs, k=self.topk)
            values, predictions = topk.values.numpy(), topk.indices.numpy()

            for v, p in zip(values.tolist(), predictions.tolist()):
                tokens = input_ids.numpy()
                tokens[masked_index] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                result.append(
                    {
                        "sequence": self.tokenizer.decode(tokens),
                        "score": v,
                        "token": p,
                        "token_str": self.tokenizer.convert_ids_to_tokens(p),
                    }
                )

            # Append
            results += [result]

        if len(results) == 1:
            return results[0]
        return results


class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using ModelForTokenClassification head. See the
    `named entity recognition usage <../usage.html#named-entity-recognition>`__ examples for more information.

    This token recognition pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "ner", for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous.

    The models that this pipeline can use are models that have been fine-tuned on a token classification task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=token-classification>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    default_input_names = "sequences"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        ignore_labels=["O"],
        task: str = "",
        grouped_entities: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task,
        )

        self.check_model_type(
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.ignore_labels = ignore_labels
        self.grouped_entities = grouped_entities

    def __call__(self, *args, **kwargs):
        inputs = self._args_parser(*args, **kwargs)
        answers = []
        for sentence in inputs:

            # Manage correct placement of the tensors
            with self.device_placement():

                tokens = self.tokenizer(
                    sentence, return_attention_mask=False, return_tensors="tf", truncation=True,
                )

                # Forward
                entities = self.model(tokens.data)[0][0].numpy()
                input_ids = tokens["input_ids"].numpy()[0]

            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)

            entities = []
            # Filter to labels not in `self.ignore_labels`
            filtered_labels_idx = [
                (idx, label_idx)
                for idx, label_idx in enumerate(labels_idx)
                if self.model.config.id2label[label_idx] not in self.ignore_labels
            ]

            for idx, label_idx in filtered_labels_idx:

                entity = {
                    "word": self.tokenizer.convert_ids_to_tokens(int(input_ids[idx])),
                    "score": score[idx][label_idx].item(),
                    "entity": self.model.config.id2label[label_idx],
                    "index": idx,
                }

                entities += [entity]

            # Append grouped entities
            if self.grouped_entities:
                answers += [self.group_entities(entities)]
            # Append ungrouped entities
            else:
                answers += [entities]

        if len(answers) == 1:
            return answers[0]
        return answers

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Returns grouped sub entities
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"]
        scores = np.mean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
        }
        return entity_group

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Returns grouped entities
        """

        entity_groups = []
        entity_group_disagg = []

        if entities:
            last_idx = entities[-1]["index"]

        for entity in entities:
            is_last_idx = entity["index"] == last_idx
            if not entity_group_disagg:
                entity_group_disagg += [entity]
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
                continue

            # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" suffixes
            if (
                entity["entity"].split("-")[-1] == entity_group_disagg[-1]["entity"].split("-")[-1]
                and entity["index"] == entity_group_disagg[-1]["index"] + 1
            ):
                entity_group_disagg += [entity]
                # Group the entities at the last entity
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
            # If the current entity is different from the previous entity, aggregate the disaggregated entity group
            else:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
                entity_group_disagg = [entity]
                # If it's the last entity, add it to the entity groups
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]

        return entity_groups


NerPipeline = TokenClassificationPipeline



# Register all the supported tasks here
SUPPORTED_TASKS = {
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "tf": TFAutoModel,
        "default": {"model": {"tf": "albert-base-v2"}},
    },
    "sentiment-analysis": {
        "impl": TextClassificationPipeline,
        "tf": TFAutoModelForSequenceClassification,
        "default": {
            "model": {
                "tf": "google/bert_uncased_L-2_H-128_A-2",
            },
        },
    },
    "ner": {
        "impl": TokenClassificationPipeline,
        "tf": TFAutoModelForTokenClassification,
        "default": {
            "model": {
                "tf": "bert-base-cased",
            },
        },
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "tf": TFAutoModelWithLMHead,
        "default": {"model": {"tf": "albert-base-v2"}},
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "tf": TFAutoModelWithLMHead,
        "default": {"model": {"tf": "albert-base-v2"}},
    },
    "zero-shot-classification": {
        "impl": ZeroShotClassificationPipeline,
        "tf": TFAutoModelForSequenceClassification,
        "default": {
            "model": {"tf": "albert-base-v2"},
            "config": {"tf": "albert-base-v2"},
            "tokenizer": {"tf": "albert-base-v2"},
        },
    },
}


def pipeline(
    task: str,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    **kwargs
) -> Pipeline:
    """
    Utility factory method to build a pipeline.

    Pipeline are made of:

        - A Tokenizer instance in charge of mapping raw textual input to token
        - A Model instance
        - Some (optional) post processing for enhancing model's output


    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - "feature-extraction": will return a :class:`~transformers.FeatureExtractionPipeline`
            - "sentiment-analysis": will return a :class:`~transformers.TextClassificationPipeline`
            - "ner": will return a :class:`~transformers.TokenClassificationPipeline`
            - "fill-mask": will return a :class:`~transformers.FillMaskPipeline`
            - "text-generation": will return a :class:`~transformers.TextGenerationPipeline`
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`, defaults to :obj:`None`):
            The model that will be used by the pipeline to make predictions. This can be :obj:`None`,
            a model identifier or an actual pre-trained model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.

            If :obj:`None`, the default for this pipeline will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`, defaults to :obj:`None`):
            The configuration that will be used by the pipeline to instantiate the model. This can be :obj:`None`,
            a model identifier or an actual pre-trained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.

            If :obj:`None`, the default for this pipeline will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`, defaults to :obj:`None`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be :obj:`None`,
            a model identifier or an actual pre-trained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.

            If :obj:`None`, the default for this pipeline will be loaded.

    Returns:
        :class:`~transformers.Pipeline`: Class inheriting from :class:`~transformers.Pipeline`, according to
        the task.

    Examples::

        from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        # Sentiment analysis pipeline
        pipeline('sentiment-analysis')

        # Question answering pipeline, specifying the checkpoint identifier
        pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

        # Named entity recognition pipeline, passing in a specific model and tokenizer
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        pipeline('ner', model=model, tokenizer=tokenizer)
    """
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    framework = "tf"

    targeted_task = SUPPORTED_TASKS[task]
    task_class, model_class = targeted_task["impl"], targeted_task[framework]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"][framework]

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        elif isinstance(config, str):
            tokenizer = config
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )

    modelcard = None
    # Try to infer modelcard from model or config name (if provided as str)
    if isinstance(model, str):
        modelcard = model
    elif isinstance(config, str):
        modelcard = config

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config)

    # Instantiate modelcard if needed
    if isinstance(modelcard, str):
        modelcard = ModelCard.from_pretrained(modelcard)

    # Instantiate model if needed
    if isinstance(model, str):
        model = model_class.from_pretrained(model, config=config)

    return task_class(model=model, tokenizer=tokenizer, modelcard=modelcard, task=task, **kwargs)
