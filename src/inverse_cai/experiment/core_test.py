import subprocess
import pytest

# Model to be used for testing the CLI.
# Corresponding API needs to be set to
# allow these tests to succeed.
TEST_API_MODEL = "openrouter/openai/gpt-4o-mini-2024-07-18"


@pytest.mark.slow
@pytest.mark.api
@pytest.mark.parametrize(
    "additional_args",
    [
        [],
        [
            "s0_added_principles_to_test=[test_principle1,test_principle2]",
            "s0_added_standard_principles_to_test=[v1,v2]",
            "annotator.fn_annotators=[{function:'inverse_cai.annotators.dummy.annotate_perfectly'}]",
        ],
        [
            # test that uses ICAI for Feedback Forensics use-case
            "s0_added_standard_principles_to_test=[v4]",
            "annotator.skip=true",
            "s0_skip_principle_generation=true",
        ],
    ],
)
def test_cli_minimal_experiment(additional_args: list[str]):
    """Test that we can run a minimal experiment end-to-end using the CLI."""
    # Run the CLI command in a subprocess
    cmd = [
        "icai-exp",
        "data_path=data/processed/example/example.csv",
        f"alg_model={TEST_API_MODEL}",
        *additional_args,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False  # Don't raise on non-zero exit
    )

    # Print output if test fails
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    # Check that the command succeeded
    assert result.returncode == 0, "CLI command failed"
