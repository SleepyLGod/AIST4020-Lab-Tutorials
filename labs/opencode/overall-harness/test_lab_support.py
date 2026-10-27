"""Offline tests. Synthetic events here are test inputs, not recorded agent runs.

Run: python -m unittest discover -s labs/opencode/overall-harness -p test_lab_support.py -v
Set OPENCODE_SOURCE_DIR to the pinned packages/opencode/src tree to check excerpts.
Temporary test directories are retained for inspection; no runtime or API is used.
"""

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import Mock, patch

from lab_support import (
    Redactor, RunResult, credential, event_summary, fresh_project, isolated_environment,
    configure_backend, package_records, prepare_lab, select_backend, write_configuration,
    run_command, save_json, show_changes, snapshot,
)
import lab_support as support

HERE = Path(__file__).resolve().parent


class SupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="opencode-offline-test-"))
        self.env = isolated_environment(self.root)

    def make_lab(self) -> support.Lab:
        """Build local test state without installing or calling OpenCode."""
        records = self.root / "records"
        records.mkdir(exist_ok=True)
        return support.Lab(self.root, self.env, Path("/synthetic/opencode"), self.root,
                           records, "1.18.34", model="course/test")

    def test_lab_run_preserves_command_session_and_no_tools_configuration(self) -> None:
        lab = self.make_lab()
        lab.enabled = True
        lab.chat_env = {**self.env, "OPENCODE_CONFIG": "no-tools.json"}
        reply = RunResult(0, False, .1, "", "", [
            {"type": "text", "part": {"text": "Synthetic reply"}, "sessionID": "test-session"},
        ])
        with patch("lab_support.run_command", return_value=reply) as run:
            self.assertIs(lab.run("Recall the order", session_id="test-session",
                                 tools=False, label="same"), reply)
            args, kwargs = run.call_args
            self.assertEqual(args[0], ["/synthetic/opencode", "run", "--pure", "--format", "json",
                                      "--agent", "build", "--model", "course/test", "--session",
                                      "test-session", "Recall the order"])
            self.assertEqual(kwargs["env"], lab.chat_env)
            self.assertEqual(kwargs["cwd"], lab.project)
            self.assertEqual(kwargs["timeout"], 300)
            first_log = kwargs["event_log"]
            lab.run("Recall the order", tools=False, label="fresh")
            self.assertNotIn("--session", run.call_args.args[0])
            self.assertNotEqual(first_log, run.call_args.kwargs["event_log"])

    def test_lab_run_disabled_and_failed_requests_are_not_retried(self) -> None:
        lab = self.make_lab()
        with patch("lab_support.run_command") as run:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                lab.run("Do not send this")
            run.assert_not_called()
        lab.enabled = True
        failed = RunResult(1, False, .1, "", "Synthetic error", [])
        with patch("lab_support.run_command", return_value=failed) as run:
            self.assertIs(lab.run("Repair"), failed)
            run.assert_called_once()
        for events in ([], [{"type": "tool_use", "part": {"tool": "read"}}]):
            with patch("lab_support.run_command", return_value=RunResult(0, False, .1, "", "", events)):
                with self.assertRaises(RuntimeError):
                    lab.run("Just answer", tools=False)

    def test_lab_check_omits_model_credentials_and_preserves_failed_test(self) -> None:
        lab = self.make_lab()
        lab.env.update(COURSE_API_KEY="synthetic-secret", OPENCODE_CONFIG="config.json")
        failed = RunResult(1, False, .1, "", "60.0 != 90.0", [])
        command = ["python3", "-m", "unittest", "-v"]
        with patch("lab_support.run_command", return_value=failed) as run:
            self.assertIs(lab.check(command), failed)
        self.assertEqual(run.call_args.args[0], command)
        self.assertNotIn("COURSE_API_KEY", run.call_args.kwargs["env"])
        self.assertNotIn("OPENCODE_CONFIG", run.call_args.kwargs["env"])
        self.assertIn("COURSE_API_KEY", lab.env)

    def test_lab_export_checks_saved_user_messages_not_assistant_guesses(self) -> None:
        lab = self.make_lab()
        lab.redact = Redactor(["synthetic-secret"])
        transcript = {"messages": [
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": "Test result first"}]},
            {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "A guessed preference"}]},
        ]}
        reply = RunResult(0, False, .1, json.dumps(transcript), "", [])
        with patch("lab_support.run_command", return_value=reply) as run, \
                patch("lab_support.show_messages"), contextlib.redirect_stdout(io.StringIO()):
            saved = lab.export("test-session", label="continued")
        self.assertEqual(run.call_args.args[0], ["/synthetic/opencode", "export", "test-session"])
        self.assertTrue(support.has_user_message(saved, "Test result first"))
        self.assertFalse(support.has_user_message(saved, "A guessed preference"))
        self.assertEqual(len(list(lab.records.glob("continued-*.session.json"))), 1)
        with patch("lab_support.run_command", return_value=RunResult(0, False, .1, "{}", "", [])), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "messages list"):
                lab.export("test-session", label="fresh")

    def test_tool_display_keeps_request_output_and_unknown_state(self) -> None:
        event = {"type": "tool_use", "part": {"tool": "bash", "state": {
            "input": {"command": "python3 -m unittest -v"}, "output": "FAILED (failures=1)"}}}
        with patch("lab_support.show_details") as display:
            support.show_tool_event(event, opened=True)
        self.assertIn("python3 -m unittest -v", display.call_args.args[1])
        self.assertIn("FAILED (failures=1)", display.call_args.args[1])
        self.assertTrue(display.call_args.kwargs["opened"])
        with patch("lab_support.show_details") as display:
            support.show_tool_event({"type": "tool_use", "part": {"tool": "read"}})
        self.assertIn("No returned output", display.call_args.args[1])

    def test_project_map_uses_actual_entries_and_marks_missing_files(self) -> None:
        (self.root / "grades.py").write_text("# test fixture")
        (self.root / "extra.csv").write_text("value\n1\n")
        with patch("lab_support.show_details") as display:
            support.show_project(self.root)
        text = display.call_args.args[1]
        self.assertIn("grades.py  program to repair", text)
        self.assertIn("extra.csv", text)
        self.assertIn("[Missing file: README.md]", text)
        self.assertNotIn("README.md  requirements", text)

    def test_action_summary_preserves_order_errors_and_missing_results(self) -> None:
        events = [
            {"type": "tool_use", "part": {"tool": "bash", "state": {
                "status": "completed", "input": {"command": "python3 -m unittest -v"},
                "output": "FAILED (failures=1)", "metadata": {"exit": 1}}}},
            {"type": "tool_use", "part": {"tool": "read", "state": {
                "input": {"filePath": "grades.py"}}}},
            {"type": "error", "error": {"message": "Synthetic API error"}},
        ]
        with patch("lab_support.show_table") as table, patch("lab_support.show_details") as details:
            support.show_tool_actions(RunResult(0, False, .1, "", "", events))
        rows = table.call_args.args[2]
        self.assertEqual([row[1] for row in rows], ["bash", "read"])
        self.assertEqual(rows[0][3], "completed; command exit 1")
        self.assertEqual(rows[0][4], "FAILED (failures=1)")
        self.assertEqual(rows[1][4], "No returned output reported")
        self.assertTrue(any(call.kwargs.get("opened") and "FAILED" in call.args[1]
                            for call in details.call_args_list))
        self.assertTrue(details.call_args.kwargs["opened"])
        with patch("lab_support.show_table") as table, contextlib.redirect_stdout(io.StringIO()) as output:
            support.show_tool_actions(RunResult(-15, True, .2, "", "", []))
        table.assert_not_called()
        self.assertIn("timed out", output.getvalue())
        self.assertIn("No tool actions", output.getvalue())

    def test_repair_comparison_does_not_fill_in_the_expected_answer(self) -> None:
        initial = RunResult(0, False, .1, "Average: 60.0\n", "")
        failed_test = RunResult(1, False, .1, "", "FAILED")
        for final, expected in ((initial, "Average: 60.0\n(Exit 0)"),
                                (RunResult(-15, True, .2, "", ""), "No output reported\n(Timed out)")):
            with patch("lab_support.show_table") as table:
                support.show_repair_comparison(initial, failed_test, final, failed_test, [], Redactor())
            rows = table.call_args.args[2]
            self.assertEqual(rows[0][2], expected)
            self.assertEqual(rows[1][2], "Failed")
            self.assertEqual(rows[2][2], "No changes")
            self.assertNotIn("90", repr(rows))

    def test_repair_comparison_distinguishes_test_outcomes(self) -> None:
        average = RunResult(0, False, .1, "Average: 60.0\n", "")
        failed = RunResult(1, False, .1, "", "FAILED")
        for code, timed_out, expected in ((0, False, "Passed"), (1, False, "Failed"),
                                          (-15, True, "Timed out"), (0, True, "Timed out")):
            with self.subTest(code=code, timed_out=timed_out):
                result = RunResult(code, timed_out, .1, "", "")
                self.assertEqual(support.test_outcome(result), expected)
                with patch("lab_support.show_table") as table:
                    support.show_repair_comparison(average, failed, average, result, [], Redactor())
                self.assertEqual(table.call_args.args[2][1], ["Test result", "Failed", expected])

    def test_session_comparison_separates_reply_from_saved_instruction(self) -> None:
        reply = RunResult(0, False, .1, "", "", [{"type": "text", "sessionID": "test-session",
                                                  "part": {"text": "Test result first"}}])
        # The assistant can say the right words without a matching saved user request.
        history = {"messages": [{"info": {"role": "assistant"}, "parts": [
            {"type": "text", "text": "Test result first"}]}]}
        with patch("lab_support.show_table") as table, patch("lab_support.show_details"):
            support.show_session_comparison(reply, RunResult(0, False, .1, "", ""),
                continued_history=history, fresh_history=None, instruction="Test result first", redact=Redactor())
        rows = table.call_args.args[2]
        self.assertEqual(rows[1][1:], ["Test result first", "No reply reported"])
        self.assertEqual(rows[2][1:], ["Not found", "No saved messages available"])

    def test_comparison_tables_escape_html_and_redact_credentials(self) -> None:
        rendered = support._table_html("<script>", ["<b>"], [["<img src=x onerror=alert(1)>"]])
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img", rendered)
        self.assertIn("&lt;img", rendered)
        self.assertIn('scope="col"', rendered)
        with patch("lab_support._table_html", wraps=support._table_html) as renderer, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            support.show_table("Test", ["Value"], [["synthetic-key"]], redact=Redactor(["synthetic-key"]))
        # Both notebook and terminal paths receive already redacted values.
        displayed = repr(renderer.call_args) if renderer.called else output.getvalue()
        self.assertNotIn("synthetic-key", displayed)
        self.assertIn("[REDACTED]", displayed)

    def test_export_can_defer_display_until_comparison(self) -> None:
        lab = self.make_lab()
        result = RunResult(0, False, .1, '{"messages": []}', "")
        with patch("lab_support.run_command", return_value=result), \
                patch("lab_support.show_messages") as messages, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(lab.export("test-session", label="fresh", show=False), {"messages": []})
        messages.assert_not_called()

    def test_model_request_timeout_depends_on_provider(self) -> None:
        selected = {"base_url": "http://127.0.0.1:1/v1", "model": "test"}
        for provider, expected in (("deepseek", 60000), ("nvidia", 60000),
                                   ("custom", 60000), ("ollama", 180000)):
            with self.subTest(provider=provider), contextlib.redirect_stdout(io.StringIO()):
                _, chat_env = write_configuration(
                    self.root, self.env, selected, provider=provider,
                    permissions={"*": "deny"}, context_tokens=16384,
                    output_tokens=2048, steps=12, redact=Redactor(),
                )
                for env in (self.env, chat_env):
                    config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
                    self.assertEqual(config["provider"]["course"]["options"]["timeout"], expected)

    def test_environment_does_not_inherit_other_secrets(self) -> None:
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "synthetic-private-value"}):
            env = isolated_environment(self.root)
        self.assertNotIn("NVIDIA_API_KEY", env)

    def test_credential_uses_environment_without_prompt_or_output(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, {"NVIDIA_NIM_API_KEY": " synthetic-private-value "}), \
                patch("lab_support.getpass.getpass") as prompt, contextlib.redirect_stdout(output):
            self.assertEqual(credential("NVIDIA_NIM_API_KEY"), "synthetic-private-value")
        prompt.assert_not_called()
        self.assertEqual(output.getvalue(), "")

    def test_fixture_fails_then_corrected_copy_passes(self) -> None:
        source_before = snapshot(HERE / "grade-report")
        project = fresh_project(HERE / "grade-report", self.root)
        args = [sys.executable, "-m", "unittest", "-v"]
        result = run_command(args, cwd=project, env=self.env, show=False, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("60.0 != 90.0", result.stderr + result.stdout)
        p = project / "grades.py"
        p.write_text(p.read_text().replace("sum(scores) / len(rows)", "sum(scores) / len(scores)"))
        fixed = run_command(args, cwd=project, env=self.env, show=False, timeout=10)
        self.assertEqual(fixed.returncode, 0, fixed.stderr)
        assertion = run_command([sys.executable, "-c", "from grades import average; assert average('scores.csv') == 90.0"],
                                cwd=project, env=self.env, show=False, timeout=10)
        self.assertEqual(assertion.returncode, 0)
        self.assertEqual(source_before, snapshot(HERE / "grade-report"))
        self.assertNotEqual(project, fresh_project(HERE / "grade-report", self.root))

    def test_events_are_streamed_redacted_and_saved(self) -> None:
        secret = "synthetic-test-credential"
        event = {"type": "tool_use", "sessionID": "synthetic-session", "part": {
            "tool": "bash", "state": {"status": "completed", "input": {"command": "test"},
                                      "output": f"FAILED {secret}", "metadata": {"exit": 1}}}}
        script = "import sys; print(" + repr(json.dumps(event)) + "); print('not-json'); print(" + repr(secret) + ", file=sys.stderr)"
        received = []
        log = self.root / "events.jsonl"
        result = run_command([sys.executable, "-c", script], cwd=self.root, env=self.env,
                             redact=Redactor([secret]), event_log=log, show=False, on_event=received.append)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.session_id, "synthetic-session")
        self.assertEqual(received, result.events)
        self.assertIn("command exit=1", event_summary(result.events[0]))
        self.assertNotIn("passed", event_summary(result.events[0]))
        for text in (result.stdout, result.stderr, log.read_text(), log.with_suffix(".process.json").read_text()):
            self.assertNotIn(secret, text)
        self.assertIn("not-json", result.stdout)
        self.assertIn("[REDACTED]", result.stdout)
        with self.assertRaises(FileExistsError):
            run_command([sys.executable, "-c", "pass"], cwd=self.root, env=self.env, event_log=log, show=False)

    def test_secret_split_across_process_chunks(self) -> None:
        secret = "synthetic-split-secret"
        script = "import sys,time; sys.stdout.write('synthetic-split-'); sys.stdout.flush(); time.sleep(.1); print('secret')"
        result = run_command([sys.executable, "-c", script], cwd=self.root, env=self.env,
                             show=False, redact=Redactor([secret]))
        self.assertEqual(result.stdout, "[REDACTED]\n")

    def test_error_event_rejects_zero_exit(self) -> None:
        result = RunResult(0, False, 0.1, "", "", [{"type": "error", "error": {"message": "synthetic failure"}}])
        with self.assertRaises(RuntimeError):
            result.require_success()

    def test_timeout_without_newline(self) -> None:
        script = "import sys,time; sys.stdout.write('unfinished'); sys.stdout.flush(); time.sleep(10)"
        result = run_command([sys.executable, "-c", script], cwd=self.root, env=self.env, show=False, timeout=.2)
        self.assertTrue(result.timed_out)
        self.assertLess(result.seconds, 5)
        self.assertEqual(result.stdout, "unfinished")
        with self.assertRaises(RuntimeError):
            result.require_success()

    def test_failing_tool_details_are_visible(self) -> None:
        event = {"type": "tool_use", "part": {"tool": "edit", "state": {"status": "error", "error": "denied"}}}
        with patch("lab_support.show_details") as display, contextlib.redirect_stdout(io.StringIO()):
            run_command([sys.executable, "-c", "print(" + repr(json.dumps(event)) + ")"],
                        cwd=self.root, env=self.env, event_log=self.root / "denied.jsonl")
        self.assertTrue(display.call_args.kwargs["opened"])

    def test_notebook_details_are_html_escaped(self) -> None:
        try:
            import IPython
        except ImportError:
            self.skipTest("IPython is unavailable; notebook HTML rendering is not checked")
        from lab_support import show_details
        with patch("IPython.get_ipython", return_value=object()), patch("IPython.display.display") as display:
            show_details("<script>", "<img src=x onerror=alert(1)>")
        rendered = display.call_args.args[0].data
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img", rendered)
        self.assertIn("&lt;img", rendered)

    def test_changes_include_new_and_protected_files(self) -> None:
        project = fresh_project(HERE / "grade-report", self.root)
        before = snapshot(project)
        (project / "extra.txt").write_text("unexpected")
        (project / "scores.csv").write_text("changed")
        with contextlib.redirect_stdout(io.StringIO()):
            changed = show_changes(before, snapshot(project), Redactor())
        self.assertEqual(changed, ["extra.txt", "scores.csv"])

    def test_json_storage_masks_credentials(self) -> None:
        path = self.root / "result.json"
        save_json(path, {"value": "synthetic-key"}, Redactor(["synthetic-key"]))
        self.assertEqual(json.loads(path.read_text()), {"value": "[REDACTED]"})

    def test_prepare_lab_creates_fresh_project_without_live_install(self) -> None:
        with patch("lab_support.install_opencode", return_value=Path("/synthetic/opencode")) as install, \
                contextlib.redirect_stdout(io.StringIO()):
            root, env, binary, project, records = prepare_lab(HERE, "1.18.34")
        install.assert_called_once_with(root, "1.18.34", env)
        self.assertEqual(binary, Path("/synthetic/opencode"))
        self.assertEqual(project.parent, root)
        self.assertTrue(records.is_dir())
        self.assertEqual(snapshot(project), snapshot(HERE / "grade-report"))

    def test_selected_provider_only_and_disabled_setup(self) -> None:
        backends = {"nvidia": {"base_url": "https://example.invalid/v1", "model": "test",
                               "key_env": "NVIDIA_NIM_API_KEY"},
                    "custom": {"base_url": "http://127.0.0.1:8000/v1", "model": "local", "key_env": ""},
                    "ollama": {"base_url": "http://127.0.0.1:11434/v1", "model": "test-local", "key_env": ""}}
        with patch("lab_support.credential", return_value="synthetic-secret") as key, \
                patch("lab_support.start_ollama") as local, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                configure_backend("nvidia", backends, root=self.root, env=self.env,
                                  context_tokens=16384, enabled=False)
            key.assert_not_called()
            local.assert_not_called()
            selected, redact = configure_backend("nvidia", backends, root=self.root, env=self.env,
                                                  context_tokens=16384, enabled=True)
            key.assert_called_once_with("NVIDIA_NIM_API_KEY")
            local.assert_not_called()
            self.assertEqual(selected["model"], "test")
            self.assertEqual(redact("synthetic-secret"), "[REDACTED]")
            configure_backend("custom", backends, root=self.root, env=self.env,
                              context_tokens=16384, enabled=True)
            self.assertEqual(key.call_count, 1)
            configure_backend("ollama", backends, root=self.root, env=self.env,
                              context_tokens=16384, enabled=True)
            local.assert_called_once_with(self.root, "test-local", 16384, self.env)
        with self.assertRaises(ValueError):
            select_backend("custom", {"custom": {"base_url": "https://user:secret@example.invalid/v1", "model": "x"}})

    def test_download_includes_only_records_and_masks_keys(self) -> None:
        records = self.root / "records"
        records.mkdir()
        name = "repair-" + "a" * 32 + ".jsonl"
        (records / name).write_text('{"text": "synthetic-secret"}\n')
        (records / "opencode.json").write_text("private config")
        (records / "unrelated.txt").write_text("private notes")
        (records / ("summary-" + "b" * 32 + ".json")).symlink_to(records / "opencode.json")
        with contextlib.redirect_stdout(io.StringIO()):
            bundle = package_records(records, Redactor(["synthetic-secret"]))
        with zipfile.ZipFile(bundle) as archive:
            self.assertEqual(archive.namelist(), [name])
            self.assertNotIn(b"synthetic-secret", archive.read(name))
            self.assertIn(b"[REDACTED]", archive.read(name))
        self.assertIn("synthetic-secret", (records / name).read_text())
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            package_records(empty, Redactor())


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads((HERE / "opencode_overall_harness.ipynb").read_text())

    def test_clean_notebook_and_python_syntax(self) -> None:
        ids = []
        for cell in self.notebook["cells"]:
            ids.append(cell["id"])
            source = "".join(cell["source"])
            if cell["cell_type"] == "code":
                ast.parse(source)
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
            else:
                self.assertEqual(sum(line.startswith("```") for line in source.splitlines()) % 2, 0)
        self.assertEqual(len(ids), len(set(ids)))
        prose = "\n".join("".join(c["source"]) for c in self.notebook["cells"])
        for forbidden in ("openclaw", "/Users/von/", "smoke", "Three Meanings", "surface"):
            self.assertNotIn(forbidden, prose.lower() if forbidden.islower() else prose)
        for part in ("### Part 1", "### Part 2", "### Part 3"):
            self.assertIn(part, prose)

    def test_project_and_independent_check_cells_run_without_a_model(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="opencode-notebook-check-"))
        project = fresh_project(HERE / "grade-report", root)
        lab = support.Lab(root, isolated_environment(root), Path("/synthetic/opencode"),
                          project, root, "1.18.34")
        cells = {c["id"]: "".join(c["source"]) for c in self.notebook["cells"]}
        context = {"lab": lab, "support": support, "sys": sys}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cells["project-files"], context)
            exec(cells["initial-test"], context)
            self.assertEqual(context["initial_average"].stdout.strip(), "Average: 60.0")
            self.assertNotEqual(context["initial_test"].returncode, 0)
            # A corrected temporary copy tests the checks, not an agent's repair ability.
            implementation = project / "grades.py"
            implementation.write_text(implementation.read_text().replace(
                "sum(scores) / len(rows)", "sum(scores) / len(scores)"))
            exec(cells["check-repair"], context)
        self.assertEqual(context["changed_files"], ["grades.py"])
        self.assertEqual(context["test_after"].returncode, 0)
        self.assertEqual(context["average_check"].returncode, 0)
        self.assertEqual(context["average_check"].stdout.strip(), "Average: 90.0")

    def test_download_without_session_comparison(self) -> None:
        cell = next(c for c in self.notebook["cells"] if c["id"] == "download")
        source = "".join(cell["source"]).replace("DOWNLOAD_RECORDS = False", "DOWNLOAD_RECORDS = True", 1)
        lab = Mock()
        # Synthetic failed run: no same/fresh conversation was created.
        result = RunResult(1, False, .1, "", "")
        context = {"lab": lab, "repair": result, "changed_files": [],
                   "test_after": result, "average_check": result}
        exec(compile(source, "download", "exec"), context)
        lab.save_record.assert_called_once_with({
            "repair_session": None, "changed_files": [],
            "your_test_passed": False, "average_is_90": False,
        })
        lab.download_records.assert_called_once_with()

    def test_config_cell_has_no_credentials_or_provider_fallback(self) -> None:
        cell = next(c for c in self.notebook["cells"] if "PERMISSIONS = {" in "".join(c["source"]))
        root = Path(tempfile.mkdtemp(prefix="opencode-config-test-"))
        records = root / "records"
        records.mkdir()
        lab = support.Lab(root, isolated_environment(root), Path("/synthetic/opencode"),
                          root, records, "1.18.34")
        context = {"lab": lab, "backend": "custom", "ENABLE_MODEL_CALLS": True,
                   "BACKENDS": {"custom": {"base_url": "http://127.0.0.1:1/v1",
                                            "model": "test", "key_env": ""}}}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile("".join(cell["source"]), "notebook-config", "exec"), context)
        cfg = json.loads((root / "opencode.json").read_text())
        self.assertEqual(cfg["provider"]["course"]["options"]["apiKey"], "{env:COURSE_API_KEY}")
        self.assertEqual(cfg["enabled_providers"], ["course"])
        chat = json.loads((root / "opencode-no-tools.json").read_text())
        self.assertEqual(chat["permission"], {"*": "deny"})
        self.assertEqual(chat["agent"]["build"]["permission"], {"*": "deny"})

    def test_backend_selection_and_disabled_calls(self) -> None:
        source = next("".join(c["source"]) for c in self.notebook["cells"]
                      if c["cell_type"] == "code" and "BACKENDS = {" in "".join(c["source"]))
        context = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, context)
        self.assertEqual(context["backend"], "ollama")
        self.assertTrue(context["ENABLE_MODEL_CALLS"])
        self.assertEqual(context["BACKENDS"]["nvidia"]["model"], "z-ai/glm-5.3-flash")
        for backend in ("deepseek", "nvidia", "ollama"):
            selected = select_backend(backend, context["BACKENDS"])
            self.assertEqual(selected["model"], context["BACKENDS"][backend]["model"])
            self.assertEqual(context["BACKENDS"]["nvidia"]["key_env"], "NVIDIA_NIM_API_KEY")
        with self.assertRaisesRegex(ValueError, "Fill in"):
            select_backend("custom", context["BACKENDS"])
        root = Path(tempfile.mkdtemp(prefix="opencode-disabled-test-"))
        lab = support.Lab(root, {}, Path("/synthetic/opencode"), root, root, "1.18.34")
        with patch("lab_support.credential") as key, patch("lab_support.start_ollama") as ollama:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                lab.configure("deepseek", context["BACKENDS"], permissions={}, enabled=False)
        key.assert_not_called()
        ollama.assert_not_called()
        self.assertFalse(lab.enabled)

    @unittest.skipUnless(os.environ.get("OPENCODE_SOURCE_DIR"), "Provide the pinned source tree to verify excerpts")
    def test_source_excerpts_match_pinned_files(self) -> None:
        root = Path(os.environ["OPENCODE_SOURCE_DIR"])
        verified_paths = set()
        for cell in self.notebook["cells"]:
            refs = cell["metadata"].get("source_excerpts", [])
            if not refs:
                continue
            blocks = re.findall(r"```ts\n(.*?)\n```", "".join(cell["source"]), re.S)
            self.assertEqual(len(blocks), len(refs))
            for block, ref in zip(blocks, refs):
                lines = (root / ref["path"]).read_text().splitlines()
                expected = [line for start, end in ref["ranges"] for line in lines[start - 1:end]]
                shown = [line for line in block.splitlines() if line.strip() != "// ..."]
                self.assertEqual([x.strip() for x in shown if x.strip()],
                                 [x.strip() for x in expected if x.strip()], ref["path"])
                repo_path = ref.get("repo_path", "packages/opencode/src/" + ref["path"])
                self.assertIn("/blob/v1.18.34/" + repo_path, "".join(cell["source"]))
                verified_paths.add(ref["path"])
        self.assertEqual(verified_paths, {
            "cli/cmd/run.ts", "session/llm.ts", "session/llm/request.ts", "session/tools.ts", "tool/read.ts",
            "session/processor.ts", "session/prompt.ts", "session/message-v2.ts",
        })


if __name__ == "__main__":
    unittest.main()
