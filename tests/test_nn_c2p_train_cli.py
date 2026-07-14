from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

from pyghl.nn_c2p import nn_c2p_train


class TrainCLIParserTests(unittest.TestCase):
    def test_eos_file_is_optional(self) -> None:
        args = nn_c2p_train.build_parser(prog="pyghl train").parse_args([])

        self.assertIsNone(args.eos_file)

    def test_explicit_eos_file_still_works(self) -> None:
        args = nn_c2p_train.build_parser(prog="pyghl train").parse_args(["local.h5"])

        self.assertEqual(str(args.eos_file), "local.h5")

    def test_missing_eos_file_uses_stellarcollapse_selection(self) -> None:
        chooser = mock.Mock(return_value=Path("downloaded.h5"))

        result = nn_c2p_train.resolve_eos_file(None, choose_and_download=chooser)

        self.assertEqual(result, Path("downloaded.h5"))
        chooser.assert_called_once_with()

    def test_explicit_eos_file_skips_stellarcollapse_selection(self) -> None:
        chooser = mock.Mock(side_effect=AssertionError("selector should not run"))

        result = nn_c2p_train.resolve_eos_file(
            Path("local.h5"), choose_and_download=chooser
        )

        self.assertEqual(result, Path("local.h5"))
        chooser.assert_not_called()

    def test_cancelled_remote_selection_returns_shell_interrupt_status(self) -> None:
        stderr = io.StringIO()

        with mock.patch.object(
            nn_c2p_train, "resolve_eos_file", side_effect=KeyboardInterrupt
        ):
            with contextlib.redirect_stderr(stderr):
                status = nn_c2p_train.main([])

        self.assertEqual(status, 130)
        self.assertIn("cancelled", stderr.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
