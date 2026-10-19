
#!/usr/bin/env python3

import argparse
import fnmatch
from pathlib import Path


def matches(path: Path, patterns: list[str], root: Path) -> bool:
    """
    Match a file against filename, relative path, and directory patterns.

    Examples:
        *.py          All Python files
        README.md     Files named README.md
        docs/         Any directory named docs and its contents
        demo*/        Any directory starting with demo
        src/tests/    The src/tests directory and its contents
        examples      Files or directories named examples
    """
    relative_path = path.relative_to(root)
    relative = relative_path.as_posix()
    parent_paths = [
        parent.as_posix()
        for parent in relative_path.parents
        if parent != Path(".")
    ]

    for raw_pattern in patterns:
        pattern = raw_pattern.replace("\\", "/")
        directory_only = pattern.endswith("/")
        pattern = pattern.rstrip("/")

        if not pattern:
            continue

        # Match the file itself, unless this is directory-only.
        if not directory_only:
            if (
                fnmatch.fnmatch(path.name, pattern)
                or fnmatch.fnmatch(relative, pattern)
            ):
                return True

        # Match any parent directory.
        for parent in parent_paths:
            parent_name = Path(parent).name

            if (
                fnmatch.fnmatch(parent_name, pattern)
                or fnmatch.fnmatch(parent, pattern)
            ):
                return True

    return False


def aggregate_files(
    root: Path,
    output: Path,
    delimiter: str = ";;;;;;",
    whitelist: list[str] | None = None,
    blacklist: list[str] | None = None,
    exclude_dirs: list[str] | None = None,
    encoding: str = "utf-8",
    include_empty: bool = False,
):
    root = root.resolve()
    output = output.resolve()

    whitelist = whitelist or []
    blacklist = blacklist or []
    exclude_dirs = exclude_dirs or []

    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    print(f"Source: {root}")
    print(f"Output: {output}")
    print(f"Delimiter: {delimiter!r}")
    print(f"Whitelist: {whitelist or '(all files)'}")
    print(f"Blacklist: {blacklist or '(none)'}")
    print(f"Excluded directories: {exclude_dirs or '(none)'}")
    print()

    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    )

    included = 0
    skipped = 0

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding=encoding) as out:
        for path in files:
            relative = path.relative_to(root)

            # Never aggregate the output file itself.
            if path.resolve() == output:
                continue

            # Exclude directory patterns.
            if exclude_dirs and matches(
                path,
                [
                    pattern.rstrip("/") + "/"
                    for pattern in exclude_dirs
                ],
                root,
            ):
                skipped += 1
                continue

            # Whitelist applies to filenames and relative paths.
            # A matching parent directory also permits its files.
            if whitelist and not matches(path, whitelist, root):
                skipped += 1
                continue

            # Blacklist takes precedence over whitelist.
            if blacklist and matches(path, blacklist, root):
                skipped += 1
                continue

            try:
                content = path.read_text(encoding=encoding)
            except (UnicodeError, OSError) as exc:
                print(f"Skipping {relative}: {exc}")
                skipped += 1
                continue

            if not include_empty and not content.strip():
                skipped += 1
                continue

            out.write(f"{delimiter}\n")
            out.write(f"`{relative.as_posix()}`\n")
            out.write(f"{delimiter}\n")
            out.write(content.rstrip("\n"))
            out.write("\n\n")

            included += 1

    print()
    print(f"Aggregated {included} files into {output}")
    print(f"Skipped {skipped} files")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recursively aggregate text files into one document, "
            "with whitelist and blacklist filtering."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Directory to recursively scan.",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("aggregated.txt"),
        help="Output file (default: aggregated.txt).",
    )

    parser.add_argument(
        "-d", "--delimiter",
        default=";;;;;;",
        help="Delimiter surrounding each filename.",
    )

    parser.add_argument(
        "-w", "--whitelist",
        nargs="+",
        action="extend",
        default=[],
        help="Allowed filename/path/directory patterns. Repeatable.",
    )

    parser.add_argument(
        "-b", "--blacklist",
        nargs="+",
        action="extend",
        default=[],
        help="Excluded filename/path/directory patterns. Repeatable.",
    )

    parser.add_argument(
        "-x", "--exclude-dirs",
        nargs="+",
        action="extend",
        default=[],
        help="Excluded directory patterns. Repeatable.",
    )

    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding (default: utf-8).",
    )

    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include empty or whitespace-only files.",
    )

    args = parser.parse_args()

    aggregate_files(
        root=args.root,
        output=args.output,
        delimiter=args.delimiter,
        whitelist=args.whitelist,
        blacklist=args.blacklist,
        exclude_dirs=args.exclude_dirs,
        encoding=args.encoding,
        include_empty=args.include_empty,
    )


if __name__ == "__main__":
    main()
