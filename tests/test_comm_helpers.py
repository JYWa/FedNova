import os
import sys
import unittest

import torch

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from comm_helpers import (
    calc_tensor_compression_stats,
    communicate,
    dequantize_flattened_tensor,
    flatten_tensors,
    quantize_flattened_tensor,
    unflatten_tensors,
)


class TestCommHelpers(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_flatten_unflatten(self):
        tensors = [torch.randn(10, 5), torch.randn(20), torch.randn(2, 3, 4)]
        flat = flatten_tensors(tensors)
        total_numel = sum(t.numel() for t in tensors)
        self.assertEqual(flat.numel(), total_numel)

        unflat = unflatten_tensors(flat, tensors)
        self.assertEqual(len(unflat), len(tensors))
        for orig, recon in zip(tensors, unflat):
            self.assertEqual(orig.shape, recon.shape)
            self.assertTrue(torch.equal(orig, recon))

    def test_quantize_dequantize_int4(self):
        for numel in [64, 128, 250, 1024]:
            flat = torch.randn(numel, dtype=torch.float32)
            q_dict = quantize_flattened_tensor(flat, nbits=4, block_size=32)
            recon = dequantize_flattened_tensor(q_dict)

            self.assertEqual(recon.shape, flat.shape)
            self.assertEqual(recon.dtype, flat.dtype)
            cos_sim = torch.nn.functional.cosine_similarity(flat, recon, dim=0)
            self.assertGreater(cos_sim.item(), 0.98)

    def test_quantize_dequantize_int8(self):
        flat = torch.randn(512, dtype=torch.float32)
        q_dict = quantize_flattened_tensor(flat, nbits=8, block_size=64)
        recon = dequantize_flattened_tensor(q_dict)

        self.assertEqual(recon.shape, flat.shape)
        cos_sim = torch.nn.functional.cosine_similarity(flat, recon, dim=0)
        self.assertGreater(cos_sim.item(), 0.999)

    def test_zero_and_empty_tensor(self):
        zero_tensor = torch.zeros(128, dtype=torch.float32)
        q_dict = quantize_flattened_tensor(zero_tensor, nbits=4, block_size=32)
        recon = dequantize_flattened_tensor(q_dict)

        self.assertTrue(torch.equal(zero_tensor, recon))
        self.assertFalse(torch.isnan(recon).any())
        self.assertFalse(torch.isinf(recon).any())

        empty_tensor = torch.empty(0, dtype=torch.float32)
        q_empty = quantize_flattened_tensor(empty_tensor, nbits=4, block_size=32)
        recon_empty = dequantize_flattened_tensor(q_empty)
        self.assertEqual(recon_empty.numel(), 0)

    def test_sub_byte_bit_packing(self):
        numel = 256
        tensor = torch.randn(numel, dtype=torch.float32)
        q_dict = quantize_flattened_tensor(tensor, nbits=4, block_size=64)

        # 256 4-bit nibbles should pack into exactly 128 uint8 bytes
        self.assertEqual(q_dict["data"].dtype, torch.uint8)
        self.assertEqual(q_dict["data"].numel(), numel // 2)

    def test_communicate_unquantized_fallback(self):
        tensors = [torch.randn(10, 5), torch.randn(20)]
        expected = [t.clone() for t in tensors]

        def mock_op(tensor):
            tensor.mul_(2.0)

        communicate(tensors, mock_op)
        for orig, updated in zip(expected, tensors):
            self.assertTrue(torch.allclose(updated, orig * 2.0))

    def test_communicate_quantized(self):
        tensors = [torch.randn(16, 8), torch.randn(16)]

        def mock_identity(tensor):
            pass

        orig_copies = [t.clone() for t in tensors]
        communicate(tensors, mock_identity, nbits=4, block_size=16)

        for orig, comm in zip(orig_copies, tensors):
            self.assertEqual(orig.shape, comm.shape)
            cos = torch.nn.functional.cosine_similarity(
                orig.flatten(), comm.flatten(), dim=0
            )
            self.assertGreater(cos.item(), 0.98)

    def test_error_feedback_residual_accumulation(self):
        tensors = [torch.randn(64)]
        residuals = torch.zeros_like(flatten_tensors(tensors))
        true_accum = torch.zeros(64)
        quant_accum = torch.zeros(64)

        for _ in range(10):
            step_delta = torch.randn(64) * 0.05
            true_accum += step_delta
            tensors[0].copy_(step_delta)

            def mock_collect(tensor):
                quant_accum.add_(tensor)

            communicate(
                tensors,
                mock_collect,
                nbits=4,
                block_size=32,
                residuals=residuals,
            )

        drift = torch.norm(true_accum - quant_accum).item()
        self.assertLess(drift, 0.05)

    def test_calc_tensor_compression_stats(self):
        tensors = [torch.randn(64, 32), torch.randn(64)]
        stats = calc_tensor_compression_stats(tensors, nbits=4, block_size=64)

        self.assertGreater(stats["compression_ratio"], 6.0)
        self.assertGreater(stats["bandwidth_savings_pct"], 80.0)
        self.assertGreater(stats["raw_bytes"], stats["compressed_bytes"])

    def test_invalid_inputs(self):
        tensor = torch.randn(32)
        with self.assertRaises(ValueError):
            quantize_flattened_tensor(tensor, nbits=7)
        with self.assertRaises(ValueError):
            quantize_flattened_tensor(tensor, nbits=4, block_size=7)


if __name__ == "__main__":
    unittest.main()
