"""
CLI smoke tests for knapsack-gnn entrypoints.
"""

from click.testing import CliRunner

from knapsack_gnn import cli


def test_cli_main_help():
    """`knapsack-gnn --help` should render usage information."""
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    assert "Knapsack GNN" in result.output


def test_cli_subcommand_helps():
    """Core subcommands should expose help text."""
    runner = CliRunner()
    for command in ["train", "eval", "ood", "pipeline", "ablation", "compare", "export", "bench"]:
        result = runner.invoke(cli.main, [command, "--help"])
        assert result.exit_code == 0, f"{command} --help should succeed"
