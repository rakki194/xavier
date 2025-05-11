import unittest
import torch
from quantization.svd_utils import decompose_weights_svd


class TestSVDUtils(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_features_tall = 128
        self.out_features_tall = 64
        self.rank_tall = 32

        self.in_features_wide = 64
        self.out_features_wide = 128
        self.rank_wide = 32

        self.in_features_square = 96
        self.out_features_square = 96
        self.rank_square = 24

        self.hat_W_tall = torch.randn(
            (self.in_features_tall, self.out_features_tall),
            dtype=torch.float32,
            device=self.device,
        )
        self.hat_W_wide = torch.randn(
            (self.in_features_wide, self.out_features_wide),
            dtype=torch.float32,
            device=self.device,
        )
        self.hat_W_square = torch.randn(
            (self.in_features_square, self.out_features_square),
            dtype=torch.float32,
            device=self.device,
        )

    def _run_decompose_test(
        self, hat_W, r, expected_in_features, expected_out_features
    ):
        L1_hp, L2_hp, R_hp = decompose_weights_svd(hat_W, r)

        # Check dtypes (should be float32 as SVD is done in float32)
        self.assertEqual(L1_hp.dtype, torch.float32)
        self.assertEqual(L2_hp.dtype, torch.float32)
        self.assertEqual(R_hp.dtype, torch.float32)

        # Check shapes
        # If r was clamped, use the clamped r for shape checking
        clamped_r = min(r, hat_W.shape[0], hat_W.shape[1])
        self.assertEqual(L1_hp.shape, (expected_in_features, clamped_r))
        self.assertEqual(L2_hp.shape, (clamped_r, expected_out_features))
        self.assertEqual(R_hp.shape, (expected_in_features, expected_out_features))

        # Check reconstruction (hat_W should be approx L1 @ L2 + R)
        reconstructed_W = (L1_hp @ L2_hp) + R_hp
        self.assertTrue(
            torch.allclose(hat_W.float(), reconstructed_W, atol=1e-5),
            f"Reconstruction failed. Max diff: {torch.max(torch.abs(hat_W.float() - reconstructed_W))}",
        )

    def test_decompose_standard_tall_matrix(self):
        self._run_decompose_test(
            self.hat_W_tall,
            self.rank_tall,
            self.in_features_tall,
            self.out_features_tall,
        )

    def test_decompose_standard_wide_matrix(self):
        self._run_decompose_test(
            self.hat_W_wide,
            self.rank_wide,
            self.in_features_wide,
            self.out_features_wide,
        )

    def test_decompose_standard_square_matrix(self):
        self._run_decompose_test(
            self.hat_W_square,
            self.rank_square,
            self.in_features_square,
            self.out_features_square,
        )

    def test_decompose_rank_one(self):
        self._run_decompose_test(
            self.hat_W_square, 1, self.in_features_square, self.out_features_square
        )

    def test_decompose_rank_clamping(self):
        # Test rank clamping if r > min(shape)
        # r will be min(self.in_features_tall, self.out_features_tall) which is self.out_features_tall (64)
        # Pass a rank larger than the smallest dimension
        requested_r = self.out_features_tall + 10
        self._run_decompose_test(
            self.hat_W_tall, requested_r, self.in_features_tall, self.out_features_tall
        )

        requested_r_wide = self.in_features_wide + 5
        self._run_decompose_test(
            self.hat_W_wide,
            requested_r_wide,
            self.in_features_wide,
            self.out_features_wide,
        )

    def test_decompose_fp16_input(self):
        hat_W_fp16 = self.hat_W_square.half()
        # Outputs L1, L2, R should still be float32 due to internal casting for SVD
        L1_hp, L2_hp, R_hp = decompose_weights_svd(hat_W_fp16, self.rank_square)
        self.assertEqual(L1_hp.dtype, torch.float32)
        self.assertEqual(L2_hp.dtype, torch.float32)
        self.assertEqual(R_hp.dtype, torch.float32)

        self.assertEqual(L1_hp.shape, (self.in_features_square, self.rank_square))
        self.assertEqual(L2_hp.shape, (self.rank_square, self.out_features_square))
        self.assertEqual(
            R_hp.shape, (self.in_features_square, self.out_features_square)
        )

        reconstructed_W = (L1_hp @ L2_hp) + R_hp
        self.assertTrue(
            torch.allclose(
                hat_W_fp16.float(), reconstructed_W, atol=1e-3
            ),  # Higher tolerance for fp16 input
            f"Reconstruction failed for fp16 input. Max diff: {torch.max(torch.abs(hat_W_fp16.float() - reconstructed_W))}",
        )

    def test_decompose_bf16_input(self):
        hat_W_bf16 = self.hat_W_square.bfloat16()
        L1_hp, L2_hp, R_hp = decompose_weights_svd(hat_W_bf16, self.rank_square)
        self.assertEqual(L1_hp.dtype, torch.float32)
        self.assertEqual(L2_hp.dtype, torch.float32)
        self.assertEqual(R_hp.dtype, torch.float32)

        reconstructed_W = (L1_hp @ L2_hp) + R_hp
        self.assertTrue(
            torch.allclose(
                hat_W_bf16.float(), reconstructed_W, atol=1e-2
            ),  # Higher tolerance for bf16 input
            f"Reconstruction failed for bf16 input. Max diff: {torch.max(torch.abs(hat_W_bf16.float() - reconstructed_W))}",
        )

    def test_decompose_svd_failure_nan_input(self):
        # torch.linalg.svd handles NaN/Inf by raising an error or returning NaNs depending on version/setup
        # Our wrapper catches LinAlgError and should return NaNs as per its logic.
        nan_hat_W = torch.full_like(self.hat_W_square, float("nan"))
        r = self.rank_square

        # Suppress print statements from the function during this test
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        L1_hp, L2_hp, R_hp = decompose_weights_svd(nan_hat_W, r)

        sys.stdout = old_stdout  # Restore stdout
        # print("Captured output from SVD failure test:", captured_output.getvalue())

        self.assertTrue(torch.all(torch.isnan(L1_hp)))
        self.assertTrue(torch.all(torch.isnan(L2_hp)))
        self.assertTrue(torch.all(torch.isnan(R_hp)))

        self.assertEqual(L1_hp.shape, (self.in_features_square, r))
        self.assertEqual(L2_hp.shape, (r, self.out_features_square))
        self.assertEqual(
            R_hp.shape, (self.in_features_square, self.out_features_square)
        )


if __name__ == "__main__":
    unittest.main()
