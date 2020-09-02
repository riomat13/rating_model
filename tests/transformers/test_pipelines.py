import unittest
from typing import Iterable, List, Optional

from rating_model.transformers import pipeline
from rating_model.transformers.pipelines import SUPPORTED_TASKS, DefaultArgumentHandler, Pipeline
from rating_model.transformers.testing_utils import slow


VALID_INPUTS = ["A simple string", ["list of strings"]]

# xlnet-base-cased disabled for now, since it crashes TF2
FEATURE_EXTRACT_FINETUNED_MODELS = ["albert-base-v2"]
TEXT_CLASSIF_FINETUNED_MODELS = ["albert-base-v2"]

FILL_MASK_FINETUNED_MODELS = ["albert-base-v2"]  # @slow

expected_fill_mask_result = [
    [
        {"sequence": "<s>My name is John</s>", "score": 0.00782308354973793, "token": 610, "token_str": "ĠJohn"},
        {"sequence": "<s>My name is Chris</s>", "score": 0.007475061342120171, "token": 1573, "token_str": "ĠChris"},
    ],
    [
        {"sequence": "<s>The largest city in France is Paris</s>", "score": 0.3185044229030609, "token": 2201},
        {"sequence": "<s>The largest city in France is Lyon</s>", "score": 0.21112334728240967, "token": 12790},
    ],
]

SUMMARIZATION_KWARGS = dict(num_beams=2, min_length=2, max_length=5)


class DefaultArgumentHandlerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = DefaultArgumentHandler()

    def test_kwargs_x(self):
        mono_data = {"X": "This is a sample input"}
        mono_args = self.handler(**mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        multi_data = {"x": ["This is a sample input", "This is a second sample input"]}
        multi_args = self.handler(**multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)

    def test_kwargs_data(self):
        mono_data = {"data": "This is a sample input"}
        mono_args = self.handler(**mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        multi_data = {"data": ["This is a sample input", "This is a second sample input"]}
        multi_args = self.handler(**multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)

    def test_multi_kwargs(self):
        mono_data = {"data": "This is a sample input", "X": "This is a sample input 2"}
        mono_args = self.handler(**mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 2)

        multi_data = {
            "data": ["This is a sample input", "This is a second sample input"],
            "test": ["This is a sample input 2", "This is a second sample input 2"],
        }
        multi_args = self.handler(**multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 4)

    def test_args(self):
        mono_data = "This is a sample input"
        mono_args = self.handler(mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        mono_data = ["This is a sample input"]
        mono_args = self.handler(mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        multi_data = ["This is a sample input", "This is a second sample input"]
        multi_args = self.handler(multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)

        multi_data = ["This is a sample input", "This is a second sample input"]
        multi_args = self.handler(*multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)


class MonoColumnInputTestCase(unittest.TestCase):
    def _test_mono_column_pipeline(
        self,
        nlp: Pipeline,
        valid_inputs: List,
        output_keys: Iterable[str],
        invalid_inputs: List = [None],
        expected_multi_result: Optional[List] = None,
        expected_check_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0], **kwargs)
        self.assertIsInstance(mono_result, list)
        self.assertIsInstance(mono_result[0], (dict, list))

        if isinstance(mono_result[0], list):
            mono_result = mono_result[0]

        for key in output_keys:
            self.assertIn(key, mono_result[0])

        multi_result = [nlp(input) for input in valid_inputs]
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], (dict, list))

        if expected_multi_result is not None:
            for result, expect in zip(multi_result, expected_multi_result):
                for key in expected_check_keys or []:
                    self.assertEqual(
                        set([o[key] for o in result]), set([o[key] for o in expect]),
                    )

        if isinstance(multi_result[0], list):
            multi_result = multi_result[0]

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

        self.assertRaises(Exception, nlp, invalid_inputs)

    def test_tf_sentiment_analysis(self):
        mandatory_keys = {"label", "score"}
        for model_name in TEXT_CLASSIF_FINETUNED_MODELS:
            nlp = pipeline(task="sentiment-analysis", model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys)

    def test_tf_feature_extraction(self):
        for model_name in FEATURE_EXTRACT_FINETUNED_MODELS:
            nlp = pipeline(task="feature-extraction", model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, {})

    @slow
    def test_tf_fill_mask_results(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        invalid_inputs = [
            "This is <mask> <mask>"  # More than 1 mask_token in the input is not supported
            "This is"  # No mask_token is not supported
        ]
        for model_name in FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs,
                mandatory_keys,
                invalid_inputs,
                expected_multi_result=expected_fill_mask_result,
                expected_check_keys=["sequence"],
            )


class ZeroShotClassificationPipelineTests(unittest.TestCase):
    def _test_scores_sum_to_one(self, result):
        sum = 0.0
        for score in result["scores"]:
            sum += score
        self.assertAlmostEqual(sum, 1.0)

    def _test_zero_shot_pipeline(self, nlp):
        output_keys = {"sequence", "labels", "scores"}
        valid_mono_inputs = [
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": "politics"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ["politics"]},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": "politics, public health"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ["politics", "public health"]},
            {"sequences": ["Who are you voting for in 2020?"], "candidate_labels": "politics"},
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "This text is about {}",
            },
        ]
        valid_multi_input = {
            "sequences": ["Who are you voting for in 2020?", "What is the capital of Spain?"],
            "candidate_labels": "politics",
        }
        invalid_inputs = [
            {"sequences": None, "candidate_labels": "politics"},
            {"sequences": "", "candidate_labels": "politics"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": None},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ""},
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": None,
            },
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "",
            },
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "Template without formatting syntax.",
            },
        ]
        self.assertIsNotNone(nlp)

        for mono_input in valid_mono_inputs:
            mono_result = nlp(**mono_input)
            self.assertIsInstance(mono_result, dict)
            if len(mono_result["labels"]) > 1:
                self._test_scores_sum_to_one(mono_result)

            for key in output_keys:
                self.assertIn(key, mono_result)

        multi_result = nlp(**valid_multi_input)
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], dict)
        self.assertEqual(len(multi_result), len(valid_multi_input["sequences"]))

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

            if len(result["labels"]) > 1:
                self._test_scores_sum_to_one(result)

        for bad_input in invalid_inputs:
            self.assertRaises(Exception, nlp, **bad_input)

    def _test_zero_shot_pipeline_outputs(self, nlp):
        inputs = [
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": ["politics", "public health", "science"],
            },
            {
                "sequences": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
                "candidate_labels": ["machine learning", "statistics", "translation", "vision"],
                "multi_class": True,
            },
        ]

        expected_outputs = [
            {
                "sequence": "Who are you voting for in 2020?",
                "labels": ["politics", "public health", "science"],
                "scores": [0.975, 0.015, 0.008],
            },
            {
                "sequence": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
                "labels": ["translation", "machine learning", "vision", "statistics"],
                "scores": [0.817, 0.712, 0.018, 0.017],
            },
        ]

        for input, expected_output in zip(inputs, expected_outputs):
            output = nlp(**input)
            for key in output:
                if key == "scores":
                    for output_score, expected_score in zip(output[key], expected_output[key]):
                        self.assertAlmostEqual(output_score, expected_score, places=2)
                else:
                    self.assertEqual(output[key], expected_output[key])

    def test_tf_zero_shot_classification(self):
        for model_name in TEXT_CLASSIF_FINETUNED_MODELS:
            nlp = pipeline(task="zero-shot-classification", model=model_name, tokenizer=model_name)
            self._test_zero_shot_pipeline(nlp)

    @slow
    def test_tf_zero_shot_outputs(self):
        nlp = pipeline(task="zero-shot-classification", model="roberta-large-mnli")
        self._test_zero_shot_pipeline_outputs(nlp)


class PipelineCommonTests(unittest.TestCase):
    pipelines = SUPPORTED_TASKS.keys()

    @slow
    def test_tf_defaults(self):
        # Test that pipelines can be correctly loaded without any argument
        for task in self.pipelines:
            with self.subTest(msg="Testing TF defaults with TF and {}".format(task)):
                pipeline(task)
