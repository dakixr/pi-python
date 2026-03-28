from __future__ import annotations

from dataclasses import dataclass

from pi_python.agent.loop import AgentResult
from pi_python.agent.models import Message
from pi_python.cli.main import build_parser, run_cli


@dataclass
class FakeAgent:
    output: str = "synthetic response"

    def run(self, prompt: str) -> AgentResult:
        return AgentResult(
            output=self.output,
            messages=[Message.user(prompt), Message.assistant(self.output)],
            iterations=1,
        )


def test_cli_runs_one_shot_prompt(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["--prompt", "hello world"])

    exit_code = run_cli(args, agent=FakeAgent())

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "synthetic response"
