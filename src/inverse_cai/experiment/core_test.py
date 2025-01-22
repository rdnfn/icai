import subprocess


def test_cli_minimal_experiment():
    """Test that we can run a minimal experiment end-to-end using the CLI."""
    # Run the CLI command in a subprocess
    cmd = [
        "icai-exp",
        "data_path=data/processed/example/example.csv",
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
