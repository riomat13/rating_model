import unittest
from os.path import dirname, exists
from pathlib import Path
from shutil import rmtree
from tempfile import NamedTemporaryFile, TemporaryDirectory

from rating_model.transformers import BertConfig, BertTokenizerFast, FeatureExtractionPipeline
from rating_model.transformers.convert_graph_to_onnx import (
    convert,
    ensure_valid_input,
    generate_identified_filename,
    infer_shapes,
    quantize,
)
from rating_model.transformers.testing_utils import slow


class FuncContiguousArgs:
    def forward(self, input_ids, token_type_ids, attention_mask):
        return None


class FuncNonContiguousArgs:
    def forward(self, input_ids, some_other_args, token_type_ids, attention_mask):
        return None


class OnnxExportTestCase(unittest.TestCase):
    MODEL_TO_TEST = ["bert-base-cased", "gpt2", "roberta-base"]

    @slow
    def test_export_tensorflow(self):
        for model in OnnxExportTestCase.MODEL_TO_TEST:
            self._test_export(model, "tf", 12)

    @slow
    def test_quantize_tf(self):
        for model in OnnxExportTestCase.MODEL_TO_TEST:
            path = self._test_export(model, "tf", 12)
            quantized_path = quantize(Path(path))

            # Ensure the actual quantized model is not bigger than the original one
            if quantized_path.stat().st_size >= Path(path).stat().st_size:
                self.fail("Quantized model is bigger than initial ONNX model")

    def _test_export(self, model, framework, opset, tokenizer=None):
        try:
            # Compute path
            with TemporaryDirectory() as tempdir:
                path = tempdir + "/model.onnx"

            # Remove folder if exists
            if exists(dirname(path)):
                rmtree(dirname(path))

                # Export
                convert(framework, model, path, opset, tokenizer)

                return path
        except Exception as e:
            self.fail(e)

    def test_infer_dynamic_axis_tf(self):
        """
        Validate the dynamic axis generated for each parameters are correct
        """
        from rating_model.transformers import TFBertModel

        model = TFBertModel(BertConfig.from_pretrained("bert-base-cased"))
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        self._test_infer_dynamic_axis(model, tokenizer, "tf")

    def _test_infer_dynamic_axis(self, model, tokenizer, framework):
        nlp = FeatureExtractionPipeline(model, tokenizer)

        variable_names = ["input_ids", "token_type_ids", "attention_mask", "output_0", "output_1"]
        input_vars, output_vars, shapes, tokens = infer_shapes(nlp, framework)

        # Assert all variables are present
        self.assertEqual(len(shapes), len(variable_names))
        self.assertTrue(all([var_name in shapes for var_name in variable_names]))
        self.assertSequenceEqual(variable_names[:3], input_vars)
        self.assertSequenceEqual(variable_names[3:], output_vars)

        # Assert inputs are {0: batch, 1: sequence}
        for var_name in ["input_ids", "token_type_ids", "attention_mask"]:
            self.assertDictEqual(shapes[var_name], {0: "batch", 1: "sequence"})

        # Assert outputs are {0: batch, 1: sequence} and {0: batch}
        self.assertDictEqual(shapes["output_0"], {0: "batch", 1: "sequence"})
        self.assertDictEqual(shapes["output_1"], {0: "batch"})

    def test_ensure_valid_input(self):
        """
        Validate parameters are correctly exported
        GPT2 has "past" parameter in the middle of input_ids, token_type_ids and attention_mask.
        ONNX doesn't support export with a dictionary, only a tuple. Thus we need to ensure we remove
        token_type_ids and attention_mask for now to not having a None tensor in the middle
        """
        # All generated args are valid
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        tokens = {"input_ids": [1, 2, 3, 4], "attention_mask": [0, 0, 0, 0], "token_type_ids": [1, 1, 1, 1]}
        ordered_input_names, inputs_args = ensure_valid_input(FuncContiguousArgs(), tokens, input_names)

        # Should have exactly the same number of args (all are valid)
        self.assertEqual(len(inputs_args), 3)

        # Should have exactly the same input names
        self.assertEqual(set(ordered_input_names), set(input_names))

        # Parameter should be reordered according to their respective place in the function:
        # (input_ids, token_type_ids, attention_mask)
        self.assertEqual(inputs_args, (tokens["input_ids"], tokens["token_type_ids"], tokens["attention_mask"]))

        # Generated args are interleaved with another args (for instance parameter "past" in GPT2)
        ordered_input_names, inputs_args = ensure_valid_input(FuncNonContiguousArgs(), tokens, input_names)

        # Should have exactly the one arg (all before the one not provided "some_other_args")
        self.assertEqual(len(inputs_args), 1)
        self.assertEqual(len(ordered_input_names), 1)

        # Should have only "input_ids"
        self.assertEqual(inputs_args[0], tokens["input_ids"])
        self.assertEqual(ordered_input_names[0], "input_ids")

    def test_generate_identified_name(self):
        generated = generate_identified_filename(Path("/home/something/my_fake_model.onnx"), "-test")
        self.assertEqual("/home/something/my_fake_model-test.onnx", generated.as_posix())
