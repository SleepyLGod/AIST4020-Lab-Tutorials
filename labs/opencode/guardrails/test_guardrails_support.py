"""Offline checks with synthetic events; these tests do not call a model."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import guardrails_support as support


HERE = Path(__file__).resolve().parent
PERMISSIONS = {
    "*": "deny", "read": "allow",
    "edit": {"*": "deny", "grades.py": "allow", "**/grades.py": "allow",
             "review.md": "allow", "**/review.md": "allow"},
    "bash": {"*": "deny", "python3 -m unittest -v": "allow"},
}


def event(tool: str, *, status: str = "completed", args: dict | None = None,
          output: str | None = None, error: str | None = None) -> dict:
    """Make explicitly synthetic CLI events for display tests."""
    state = {"status": status, "input": args or {}}
    if output is not None:
        state["output"] = output
    if error is not None:
        state["error"] = error
    return {"type": "tool_use", "sessionID": "synthetic-session", "part": {"tool": tool, "state": state}}


class GuardrailsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="guardrails-offline-"))
        env = support.base.isolated_environment(self.root)
        self.lab = support.Experiment(HERE, self.root, Path("/not-executed/opencode"), self.root, env, "1.18.34")
        self.lab.selected = {"base_url": "https://example.invalid/v1", "model": "synthetic-model"}
        self.lab.env["COURSE_API_KEY"] = "synthetic-test-key"
        self.lab.backend = "deepseek"
        self.lab.redact = support.base.Redactor(["synthetic-test-key"])

    def case(self, name: str = "Synthetic case", **kwargs: object) -> support.Case:
        with contextlib.redirect_stdout(io.StringIO()):
            return self.lab.new_case(name, copy.deepcopy(PERMISSIONS), **kwargs)

    def set_result(self, case: support.Case, events: list[dict]) -> support.RunResult:
        case.result = support.RunResult(0, False, 0.1, "", "", events)
        case.after = support.base.snapshot(case.project)
        return case.result

    def test_new_cases_are_pristine_and_have_separate_state(self) -> None:
        first = self.case("A")
        (first.project / "grades.py").write_text("# local synthetic change\n")
        second = self.case("B")
        self.assertNotEqual(first.project, second.project)
        self.assertNotEqual(first.env["HOME"], second.env["HOME"])
        self.assertEqual(first.before, second.before)
        self.assertIn("len(rows)", (second.project / "grades.py").read_text())
        self.assertNotIn("DEEPSEEK_API_KEY", second.env)

    def test_configuration_keeps_permissions_and_uses_environment_key_reference(self) -> None:
        case = self.case()
        config_text = Path(case.env["OPENCODE_CONFIG"]).read_text()
        config = json.loads(config_text)
        self.assertEqual(config["permission"], PERMISSIONS)
        self.assertEqual(config["agent"]["build"]["steps"], 12)
        self.assertEqual(config["provider"]["course"]["options"]["apiKey"], "{env:COURSE_API_KEY}")
        self.assertNotIn("synthetic-test-key", config_text)

    def test_disabled_setup_does_not_read_credentials(self) -> None:
        self.lab.selected = {}
        with patch.object(support.base, "credential") as credential, patch.object(support.base, "start_ollama") as ollama:
            with self.assertRaises(RuntimeError):
                self.lab.configure("deepseek", {"deepseek": self.lab.selected}, enabled=False)
        credential.assert_not_called()
        ollama.assert_not_called()

    def test_model_cannot_change_halfway_through_comparison(self) -> None:
        self.case()
        with self.assertRaisesRegex(RuntimeError, "Start a new experiment"):
            self.lab.configure("custom", {}, enabled=True)

    def test_skill_copy_does_not_silently_grant_permission(self) -> None:
        case = self.case(with_skill=True)
        source = HERE / "skills" / "repair-check" / "SKILL.md"
        self.assertEqual(source.read_bytes(), (case.project / ".opencode/skills/repair-check/SKILL.md").read_bytes())
        config = json.loads(Path(case.env["OPENCODE_CONFIG"]).read_text())
        self.assertNotIn("skill", config["permission"])

    def test_run_uses_real_cli_command_and_does_not_retry(self) -> None:
        case = self.case()
        failed = support.RunResult(-15, True, 300, "", "Synthetic timeout", [])
        with patch.object(support.base, "run_command", return_value=failed) as run, \
                patch.object(support, "show_details"), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertIs(case.run("Synthetic task"), failed)
            with self.assertRaisesRegex(RuntimeError, "Create a new case"):
                case.run("Synthetic task")
        run.assert_called_once()
        args = run.call_args.args[0]
        self.assertIn("run", args)
        self.assertIn("--pure", args)
        self.assertEqual(args[-1], "Synthetic task")
        self.assertNotIn("--session", args)
        self.assertEqual(run.call_args.kwargs["timeout"], 300)
        self.assertEqual(run.call_args.kwargs["cwd"], case.project)
        self.assertIn("helper stops after 5 minutes", output.getvalue())
        self.assertIn("asks the model to finish from loop round 12", output.getvalue())

    def test_independent_check_omits_credentials_and_keeps_failure(self) -> None:
        case = self.case()
        failed = support.RunResult(1, False, 0.1, "", "Synthetic failed test")
        with patch.object(support.base, "run_command", return_value=failed) as run:
            self.assertIs(case.check(["python3", "-m", "unittest", "-v"]), failed)
        env = run.call_args.kwargs["env"]
        self.assertNotIn("COURSE_API_KEY", env)
        self.assertNotIn("OPENCODE_CONFIG", env)
        self.assertIn("COURSE_API_KEY", case.env)

    def test_target_permission_denial_is_not_an_unrelated_command_error(self) -> None:
        case = self.case()
        denied = "The user has specified a rule which prevents you from using this specific tool call."
        self.set_result(case, [event("bash", status="error", args={"command": "ls -la"}, error=denied)])
        check = support.RunResult(1, False, 0.1, "Average: 60.0", "")
        self.assertEqual(support.observations(case, check, check)["Edit/write requests for grades.py"], "No edit/write request recorded")
        case.result.events.append(event("edit", status="error", args={"filePath": str(case.project / "grades.py")}, error=denied))
        values = support.observations(case, check, check)
        self.assertEqual(values["Edit/write requests for grades.py"], "Permission denied")
        self.assertEqual(values["grades.py changed"], "No")
        self.assertEqual(values["Your test"], "Failed")
        self.assertIn("60.0", values["Your average check"])
        self.assertNotIn("90", values["Your average check"])

    def test_generic_edit_error_is_not_reported_as_permission_denial(self) -> None:
        case = self.case()
        self.set_result(case, [event("edit", status="error", args={"filePath": "grades.py"}, error="Old text not found")])
        check = support.RunResult(1, False, 0.1, "", "")
        self.assertEqual(support.observations(case, check, check)["Edit/write requests for grades.py"], "error")

    def test_no_output_is_not_replaced_with_expected_success(self) -> None:
        case = self.case()
        self.set_result(case, [])
        check = support.RunResult(-15, True, 0.1, "", "")
        values = support.observations(case, check, check)
        self.assertEqual(values["review.md changed"], "No")
        self.assertEqual(values["Your test"], "Timed out")
        self.assertEqual(values["Your average check"], "No output (timed out)")

    def test_only_exact_new_generated_ignore_is_separated(self) -> None:
        case = self.case(with_skill=True)
        ignore = case.project / ".opencode/.gitignore"
        ignore.write_text(support._GENERATED_IGNORE)
        result = self.set_result(case, [])
        changed, setup = support.file_changes(case.before, case.after, result, case.project)
        self.assertEqual(setup, [".opencode/.gitignore"])
        self.assertNotIn(".opencode/.gitignore", changed)
        ignore.write_text("Unexpected content")
        self.set_result(case, [])
        changed, setup = support.file_changes(case.before, case.after, case.result, case.project)
        self.assertEqual(setup, [])
        self.assertIn(".opencode/.gitignore", changed)

    def test_recorded_tool_change_to_ignore_is_not_classified_as_setup(self) -> None:
        case = self.case(with_skill=True)
        (case.project / ".opencode/.gitignore").write_text(support._GENERATED_IGNORE)
        result = self.set_result(case, [event("write", args={"filePath": ".opencode/.gitignore"})])
        changed, setup = support.file_changes(case.before, case.after, result, case.project)
        self.assertIn(".opencode/.gitignore", changed)
        self.assertEqual(setup, [])

    def test_skill_evidence_comes_from_events_not_fixture(self) -> None:
        case = self.case(with_skill=True)
        self.set_result(case, [])
        with contextlib.redirect_stdout(io.StringIO()) as output, patch.object(support.base, "show_tool_event") as display:
            support.show_skill(case)
        self.assertIn("No skill-tool call", output.getvalue())
        display.assert_not_called()
        case.result.events.append(event("skill", args={"name": "repair-check"}, output="Synthetic loaded instructions"))
        with patch.object(support.base, "show_tool_event") as display:
            support.show_skill(case)
        self.assertEqual(display.call_args.args[0]["part"]["state"]["output"], "Synthetic loaded instructions")

    def test_report_text_and_errors_are_shown_from_actual_records(self) -> None:
        case = self.case()
        (case.project / "review.md").write_text("Synthetic report: no repair was completed.")
        self.set_result(case, [{"type": "error", "error": {"message": "Synthetic API error"}}])
        check = support.RunResult(1, False, 0.1, "Average: 60.0", "")
        with patch.object(support.base, "show_table"), patch.object(support.base, "show_changes"), \
                patch.object(support, "show_details") as display:
            support.show_case(case, check, check)
        texts = [str(call.args) for call in display.call_args_list]
        self.assertTrue(any("Synthetic API error" in text for text in texts))
        self.assertTrue(any("no repair was completed" in text for text in texts))

    def test_completed_report_writes_expand_without_expanding_reads(self) -> None:
        case = self.case()
        self.set_result(case, [
            event("edit", status="error", args={"filePath": "grades.py"},
                  error="The user has specified a rule which prevents this edit."),
            event("write", args={"filePath": str(case.project / "review.md")}),
            event("edit", args={"filePath": "review.md"}),
            event("read", args={"filePath": "review.md"}),
            event("write", args={"filePath": "other.md"}),
        ])
        check = support.RunResult(1, False, 0.1, "Average: 60.0", "")
        with patch.object(support.base, "show_table"), patch.object(support.base, "show_changes"), \
                patch.object(support, "show_details"), patch.object(support.base, "show_tool_event") as display:
            support.show_case(case, check, check)
        self.assertEqual([call.kwargs["opened"] for call in display.call_args_list],
                         [True, True, True, False, False])

    def test_summary_keeps_file_checks_and_moves_metadata_to_details(self) -> None:
        case = self.case(with_skill=True)
        (case.project / ".opencode/.gitignore").write_text(support._GENERATED_IGNORE)
        (case.project / "unexpected.txt").write_text("Synthetic unexpected change")
        self.set_result(case, [])
        check = support.RunResult(1, False, 0.1, "Average: 60.0", "")
        with patch.object(support.base, "show_table") as table, \
                patch.object(support.base, "show_changes") as changes, \
                patch.object(support, "show_details") as details:
            support.show_case(case, check, check)
        rows = dict(table.call_args.args[2])
        self.assertEqual(next(iter(rows)), "Edit/write requests for grades.py")
        self.assertNotIn("Session", rows)
        self.assertNotIn("OpenCode setup file", rows)
        self.assertEqual(rows["Other file changes"], "unexpected.txt")
        changes.assert_called_once_with(case.before, case.after, case.redact)
        detail = next(call for call in details.call_args_list if "run details" in call.args[0])
        self.assertIn("Session:", detail.args[1])
        self.assertIn(".opencode/.gitignore", detail.args[1])
        self.assertFalse(detail.kwargs.get("opened", False))
        saved = json.loads((case.records / "observations.json").read_text())
        self.assertIn("Session", saved)
        self.assertEqual(saved["OpenCode setup file"], ".opencode/.gitignore")

    def test_comparison_moves_metadata_but_preserves_actual_outcomes(self) -> None:
        first, second = self.case("Allowed"), self.case("Denied")
        for case in (first, second):
            self.set_result(case, [])
        failed = support.RunResult(1, False, 0.1, "Average: 60.0", "")
        with patch.object(support.base, "show_table") as table, patch.object(support, "show_details") as details:
            support.compare_cases(first, second, allowed_test=failed, denied_test=failed,
                                  allowed_average=failed, denied_average=failed)
        rows = {row[0]: row[1:] for row in table.call_args.args[2]}
        self.assertNotIn("Session", rows)
        self.assertNotIn("OpenCode setup file", rows)
        self.assertEqual(rows["grades.py changed"], ["No", "No"])
        self.assertEqual(rows["Your average check"], ["Average: 60.0 (exit 1)"] * 2)
        self.assertEqual(details.call_count, 2)

    def test_dependency_changes_are_folded_without_hiding_course_files_or_records(self) -> None:
        case = self.case(with_skill=True)
        package = case.project / ".opencode/node_modules/example"
        package.mkdir(parents=True)
        (package / "index.js").write_text("// Synthetic dependency\n")
        (case.project / ".opencode/package.json").write_text('{"private": true}\n')
        (case.project / ".opencode/unexpected.txt").write_text("Synthetic unexpected change")
        (case.project / ".opencode/skills/repair-check/SKILL.md").write_text("Synthetic changed Skill")
        (case.project / "grades.py").write_text("# Synthetic changed implementation\n")
        (case.project / "review.md").write_text("Synthetic report")
        self.set_result(case, [])
        before, after = copy.deepcopy(case.before), copy.deepcopy(case.after)
        # The run saves these full snapshots before any display filtering.
        support.base.save_json(case.records / "files.json", {"before": before, "after": after}, case.redact)
        check = support.RunResult(1, False, 0.1, "Average: 60.0", "")
        with patch.object(support.base, "show_table") as table, \
                patch.object(support.base, "show_changes") as changes, \
                patch.object(support, "show_details") as details:
            support.show_case(case, check, check)
        rows = dict(table.call_args.args[2])
        self.assertIn("4 paths", rows["Dependency changes"])
        self.assertNotIn("node_modules", rows["Other file changes"])
        self.assertIn(".opencode/unexpected.txt", rows["Other file changes"])
        self.assertIn(".opencode/skills/repair-check/SKILL.md", rows["Other file changes"])
        displayed_after = changes.call_args.args[1]
        self.assertIn("grades.py", displayed_after)
        self.assertIn("review.md", displayed_after)
        self.assertIn(".opencode/skills/repair-check/SKILL.md", displayed_after)
        self.assertNotIn(".opencode/node_modules/example/index.js", displayed_after)
        folded = next(call for call in details.call_args_list if call.args[0].startswith("Dependency changes:"))
        self.assertFalse(folded.kwargs["opened"])
        self.assertIn(".opencode/node_modules/example/index.js", folded.args[1])
        self.assertEqual(case.before, before)
        self.assertEqual(case.after, after)
        saved = json.loads((case.records / "files.json").read_text())
        self.assertEqual(saved, {"before": before, "after": after})
        with patch.object(support.base, "show_table") as comparison, patch.object(support, "show_details"):
            support.compare_cases(case, case, allowed_test=check, denied_test=check,
                                  allowed_average=check, denied_average=check)
        comparison_rows = {row[0]: row[1:] for row in comparison.call_args.args[2]}
        self.assertEqual(comparison_rows["Dependency changes"], [rows["Dependency changes"]] * 2)

    def test_notebook_case_cells_connect_with_synthetic_command_results(self) -> None:
        """Check notebook names and arguments without installing OpenCode or calling a model."""
        notebook = json.loads((HERE / "opencode_guardrails.ipynb").read_text())
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        namespace = {"lab": self.lab, "support": support, "copy": copy}
        failed = support.RunResult(1, False, 0.1, "Average: 60.0", "Synthetic check: 60.0 != 90.0", [])
        sequence = ["initial-checks", "permissions", "allow-run", "allow-check",
                    "deny-setup", "deny-check", "comparison", "skill-run"]
        with patch.object(support.base, "run_command", return_value=failed) as run, \
                patch.object(support.base, "show_table"), patch.object(support.base, "show_changes"), \
                patch.object(support, "show_details"), contextlib.redirect_stdout(io.StringIO()):
            for cell_id in sequence:
                source = cells[cell_id]
                if cell_id == "skill-run":
                    source = source.replace("RUN_SKILL_EXAMPLE = False", "RUN_SKILL_EXAMPLE = True")
                exec(compile(source, f"notebook:{cell_id}", "exec"), namespace)
        allowed_case = namespace["edit_allowed_case"]
        denied_case = namespace["edit_denied_case"]
        skill_case = namespace["skill_case"]
        self.assertEqual(allowed_case.before, denied_case.before)
        allowed_rules = json.loads(Path(allowed_case.env["OPENCODE_CONFIG"]).read_text())["permission"]
        denied_rules = json.loads(Path(denied_case.env["OPENCODE_CONFIG"]).read_text())["permission"]
        expected = copy.deepcopy(allowed_rules)
        expected["edit"]["grades.py"] = "deny"
        expected["edit"]["**/grades.py"] = "deny"
        self.assertEqual(denied_rules, expected)
        self.assertEqual(denied_rules["edit"]["review.md"], "allow")
        self.assertEqual(skill_case.name, "With repair Skill")
        agent_commands = [call.args[0] for call in run.call_args_list
                          if call.args[0][0] == str(self.lab.binary)]
        self.assertEqual(len(agent_commands), 3)
        self.assertEqual(agent_commands[0][-1], agent_commands[1][-1])
        self.assertTrue(agent_commands[2][-1].startswith("Load the repair-check skill"))

    def test_fixture_is_broken_and_minimal_fix_passes(self) -> None:
        project = self.root / "fixture-check"
        shutil.copytree(HERE / "grade-report", project)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
        command = [sys.executable, "-m", "unittest", "-v"]
        before = subprocess.run(command, cwd=project, env=env, text=True, capture_output=True, timeout=20)
        self.assertNotEqual(before.returncode, 0)
        self.assertIn("60.0 != 90.0", before.stdout + before.stderr)
        implementation = project / "grades.py"
        implementation.write_text(implementation.read_text().replace("len(rows)", "len(scores)"))
        after = subprocess.run(command, cwd=project, env=env, text=True, capture_output=True, timeout=20)
        self.assertEqual(after.returncode, 0, after.stderr)
        self.assertIn("len(rows)", (HERE / "grade-report/grades.py").read_text())


if __name__ == "__main__":
    unittest.main()
