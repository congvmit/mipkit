import argparse
import math
import os
import shlex
import subprocess
import sys
import textwrap
from functools import wraps
from subprocess import PIPE
from typing import Callable, Iterable, Iterator, List, Mapping, Optional, Tuple

TARGET_VERSION = f"py{sys.version_info.major}{sys.version_info.minor}"

HELP_WIDTH = 55


def round_up_to(x: int, base: int) -> int:
    """Round ``x`` up to the nearest multiple of ``base``."""
    return int(math.ceil(x / base)) * base


class FormattedHelpArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser that adds formatting to help text.

    Adds the following behavior to ``argparse.ArgumentParser``:

    - Uses ``argparse.RawTextHelpFormatter`` by default, but still wraps help text to 79 chars.
    - Automatically adds information about argument defaults to help text.
    - Generates custom help text for arguments with ``choices``.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", argparse.RawTextHelpFormatter)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _fill(text: str, width=HELP_WIDTH, **kwargs):
        """Calls ``textwrap.fill`` with a default width of ``HELP_WIDTH``."""
        return textwrap.fill(text, width=width, **kwargs)

    def add_argument(self, *name_or_flags: str, envvar: Optional[str] = None, **kwargs):
        """Adds some functionality to ``add_argument``.

        - Optionally take an environment variable as a default.
        - Add any default to the help text.
        - Wrap ``help`` (``argparse.RawTextHelpFormatter`` doesn't auto-wrap help text).

        :param envvar:
            Name of an environment variable to use as the default value.

            If a ``default`` is also given, it will be used as a fallback if the env var isn't set.
            If no ``default`` is given _and_ the env var isn't set, a ValueError will be raised.

            If the argument stores a list of values, the environment variable will be parsed into a
            list using `shlex.split`.
        """
        # If `envvar` given, set argument `default` to its value if set.
        if envvar:
            val = os.getenv(envvar)
            if val:
                type_func = kwargs.get("type") or str
                nargs = kwargs.get("nargs")
                # If argument stores a list, parse envvar as a list.
                if (
                    nargs
                    and (nargs in ("*", "+") or isinstance(nargs, int))
                    or kwargs.get("action") in ("append", "append_const", "extend")
                ):
                    kwargs["default"] = [type_func(item) for item in shlex.split(val)]
                else:
                    kwargs["default"] = type_func(val)
            elif "default" not in kwargs:
                raise ValueError(f"`envvar` ${envvar} not found, and no `default` was given")

        # Add default to help text.
        help_text = kwargs.get("help")
        if help_text:
            default = kwargs.get("default")

            default_list = []
            if envvar:
                default_list.append(f"${envvar}")
            if default and default is not argparse.SUPPRESS:
                default_list.append(f"{default!r}")

            if default_list:
                default_str = " | ".join(default_list)
                if "%(default)" in help_text:
                    help_text = help_text % {"default": default_str}
                else:
                    help_text += f" (default: {default_str})"

            # Wrap help text.
            kwargs["help"] = self._fill(help_text)

        return super().add_argument(*name_or_flags, **kwargs)

    def add_choices_argument(self, *name_or_flags: str, choices: Mapping[str, str], **kwargs):
        """Add an argument with ``choices``.

        The ``choices`` param takes a mapping of choices to help text for each choice, and appends
        a formatted line to the given ``help`` for each choice.

        Note that this will not work if ``formatter_class`` is set to anything other than
        ``argparse.RawTextHelpFormatter`` (the default).
        """
        choices = dict(choices)

        # If ``default`` given, prepend "(default) " to the help text of the default choice.
        default = kwargs.get("default")
        if default and default is not argparse.SUPPRESS and default in choices:
            choices[default] = f"(default) {choices[default]}"

        # Generate help text for choices.
        prefix = "> "
        max_choice_len = max(len(choice) for choice in choices.keys()) + 2
        choice_width = round_up_to(max_choice_len, 2)
        choice_help_width = HELP_WIDTH - choice_width - len(prefix)
        choice_help_indent = " " * (choice_width + len(prefix))
        choice_fmt = "{prefix}{choice:<%d}{help:<%d}" % (
            choice_width,
            choice_help_width,
        )
        choices_help = "\n".join(
            choice_fmt.format(
                prefix=prefix,
                choice=choice,
                help=self._fill(
                    text, width=choice_help_width, subsequent_indent=choice_help_indent
                ),
            )
            for choice, text in choices.items()
        )

        # Format final help text.
        help_text = "{}:\n{}".format(
            self._fill(kwargs.pop("help", "choices").rstrip(",.:;")), choices_help
        )
        return super().add_argument(*name_or_flags, choices=choices, help=help_text, **kwargs)


def select_staged(paths: Iterable[str]) -> Iterator[str]:
    yield from (file for code, file in _iter_changed(paths) if code.index_has_changes())


def select_modified(paths: Iterable[str]) -> Iterator[str]:
    yield from (
        file for code, file in _iter_changed(paths) if code.has_changes() or code.is_untracked()
    )


def select_head(paths: Iterable[str]) -> Iterator[str]:
    yield from _iter_committed(paths, "HEAD^1..HEAD")


def select_local(paths: Iterable[str]) -> Iterator[str]:
    try:
        yield from _iter_committed(paths, "@{upstream}..")
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 128:
            print(
                "pyfmt: no upstream branch: falling back to `--select all`",
                file=sys.stderr,
            )
            yield from paths
        else:
            raise


def select_all(paths: Iterable[str]) -> Iterator[str]:
    yield from paths


class GitStatusCode:
    """Wrapper around the 2-character status codes returned by ``git status --porcelain``.

    :param index: The first character, representing the file's status in the index.
    :param work_tree: The second character, representing the file's status in the working tree.
    """

    def __init__(self, index: str, work_tree: str):
        self.index = index
        self.work_tree = work_tree

    def index_has_changes(self) -> bool:
        return self.index in "MARC"

    def has_changes(self) -> bool:
        return self.index_has_changes() or self.work_tree in "MAC"

    def is_untracked(self) -> bool:
        return self.index == self.work_tree == "?"

    def is_deleted(self) -> bool:
        return self.work_tree == "D"

    def is_renamed(self) -> bool:
        return self.index == "R"


def _iter_changed(paths: Iterable[str]) -> Iterator[Tuple[GitStatusCode, str]]:
    """Iterate over .py files in the index and working tree that aren't deleted."""
    output = _sh("git", "status", "--porcelain", *paths)
    for line in output.splitlines():
        xy, line = line[:2], line[2:].strip()
        code = GitStatusCode(*xy)
        if code.is_renamed():
            _, _, file = line.split()
        else:
            file = line.strip()
        if not code.is_deleted() and file.endswith(".py"):
            yield code, file


def _iter_committed(paths: Iterable[str], commits: str) -> Iterator[str]:
    """Iterate over .py files in the given commit ("x") or range ("x..y")."""
    output = _sh("git", "--no-pager", "diff", "--numstat", commits, "--", *paths)
    for line in output.splitlines():
        file = line.strip().rsplit(maxsplit=1)[-1]
        if file.endswith(".py"):
            yield file


def _sh(*args: str) -> str:
    return subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    ).stdout.decode()


SELECT_CHOICES = {
    "all": "all files",
    "staged": "files in the index",
    "modified": "files in the index, working tree, and untracked files",
    "head": "files changed in HEAD",
    "local": "files changed locally but not upstream;"
    " if upstream branch is missing, fallback to `all`",
}

COMMIT_CHOICES = {
    "patch": "commit files with --patch",
    "amend": "commit files with --amend",
    "all": "commit all selected files, whether or not they were formatted",
}


ISORT_CMD = [
    "isort",
    "--force-grid-wrap=0",
    "--line-width={line_length}",
    "--multi-line=3",
    "--use-parentheses",
    # "--recursive",
    "--trailing-comma",
    "{extra_isort_args}",
    "{path}",
]
BLACK_CMD = [
    "black",
    "--line-length={line_length}",
    f"--target-version={TARGET_VERSION}",
    "{extra_black_args}",
    "{path}",
]

PYINK_CMD = [
    "pyink",
    "--line-length={line_length}",
    f"--target-version={TARGET_VERSION}",
    "{extra_pyink_args}",
    "{path}",
]

SELECTOR_MAP = {
    "staged": select_staged,
    "modified": select_modified,
    "head": select_head,
    "local": select_local,
    "all": select_all,
}


def fmt(
    paths: List[str],
    selector: str = "all",
    line_length: int = 100,
    check: bool = False,
    commit: Optional[List[str]] = None,
    commit_msg: Optional[str] = None,
    extra_isort_args: str = "",
    extra_pyink_args: str = "",
) -> int:
    """Run isort and black with the given params and print the results."""
    # Filter path according to given ``selector``.
    select_files = SELECTOR_MAP[selector]
    files = tuple(select_files(paths))
    if not files:
        print("No files were selected.")
        return 0

    if check:
        extra_isort_args += " --check-only"
        extra_pyink_args += " --check"

    # Run isort and pyink.
    files_str = " ".join(files)
    isort_lines, isort_exitcode = run_formatter(
        ISORT_CMD, files_str, line_length=line_length, extra_isort_args=extra_isort_args
    )
    # black_lines, black_exitcode = run_formatter(
    #     BLACK_CMD, files_str, line_length=line_length, extra_black_args=extra_black_args
    # )
    pyink_lines, pyink_exitcode = run_formatter(
        PYINK_CMD, files_str, line_length=line_length, extra_pyink_args=extra_pyink_args
    )
    exitcode = isort_exitcode or pyink_exitcode

    # Commit changes if successful.
    if not exitcode and not check and commit is not None:
        cmd = ["git", "commit"]

        if "patch" in commit:
            cmd.append("--patch")
        if "amend" in commit:
            cmd.append("--amend")

        if commit_msg is not None:
            # If no message given, use auto-commit behavior.
            if commit_msg == "":
                # If `amend` given, the commit already has a message, so just skip the editor.
                if "amend" in commit:
                    cmd.append("--no-edit")
                # Otherwise, copy the previous commit's message.
                else:
                    cmd.append("--reuse-message=HEAD")
            else:
                cmd.append(f"--message={commit_msg}")

        # If `all` given, commit all selected files. Otherwise, commit only formatted files.
        if "all" in commit:
            cmd.extend(files)
        else:
            formatted_files = {line.split()[-1] for line in isort_lines + pyink_lines}
            cmd.extend(formatted_files)

        subprocess.run(cmd)

    return exitcode


def run_formatter(cmd, path, **kwargs) -> Tuple[List[str], int]:
    """Helper to run a shell command and print prettified output."""
    cmd = shlex.split(" ".join(cmd).format(path=path, **kwargs))
    result = subprocess.run(cmd, stdout=PIPE, stderr=PIPE)

    prefix = f"{cmd[0]}: "
    sep = "\n" + (" " * len(prefix))
    lines = result.stdout.decode().splitlines() + result.stderr.decode().splitlines()

    # Remove fluff from black's output.
    if cmd[0] == "black" and result.returncode == 0:
        lines = lines[:-2]

    if "".join(lines) == "":
        print(f"{prefix}No changes.")
    else:
        print(f"{prefix}{sep.join(lines)}")

    return lines, result.returncode


def main():
    parser = FormattedHelpArgumentParser(prog="fmt")
    parser.add_argument(
        "path",
        nargs="*",
        envvar="BASE_CODE_DIR",
        default=["."],
        metavar="PATH",
        help="file and directory paths where pyfmt will be run",
    )
    parser.add_choices_argument(
        "-x",
        "--select",
        choices=SELECT_CHOICES,
        default="all",
        metavar="SELECT",
        help="filter which files to format in PATH:",
    )
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="don't write changes, just print the files that would be formatted",
    )
    parser.add_argument(
        "--line-length",
        type=int,
        envvar="MAX_LINE_LENGTH",
        default=100,
        metavar="N",
        help="max characters per line",
    )
    parser.add_choices_argument(
        "--commit",
        choices=COMMIT_CHOICES,
        nargs="*",
        metavar="ARG",
        help="commit files that were formatted. one or more args can be given to change this"
        " behavior:",
    )
    parser.add_argument(
        "--commit-msg",
        nargs="*",
        metavar="MSG",
        help="auto-commit changes. if args are given, they are concatenated to form the commit"
        " message. otherwise the current commit's log message is reused. if --commit is not"
        " present, a naked `--commit` is implied.",
    )
    parser.add_argument(
        "--extra-isort-args",
        default="",
        metavar="ARGS",
        help="additional args to pass to isort",
    )
    parser.add_argument(
        "--extra-black-args",
        default="",
        metavar="ARGS",
        help="additional args to pass to black",
    )

    parser.add_argument(
        "--extra-pyink-args",
        default="",
        metavar="ARGS",
        help="additional args to pass to pyink",
    )

    opts = parser.parse_args()

    if opts.commit_msg is not None:
        # Concatenate --commit-msg.
        opts.commit_msg = " ".join(opts.commit_msg)
        # Add implicit --commit if --commit-msg is given.
        if opts.commit is None:
            opts.commit = []

    exitcode = fmt(
        opts.path,
        opts.select,
        check=opts.check,
        line_length=opts.line_length,
        commit=opts.commit,
        commit_msg=opts.commit_msg,
        extra_isort_args=opts.extra_isort_args,
        extra_pyink_args=opts.extra_pyink_args,
    )
    sys.exit(exitcode)


if __name__ == "__main__":
    main()
