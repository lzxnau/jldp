"""
Pip Package Upgrade Python Script.

JLDP Project.
:Author:  JLDP
:Version:    2023.11.30.3
"""
import os
import sys

NAME = os.environ.get("VENV")
PNAME = NAME + ".txt" if NAME else sys.exit()
UNAME = os.environ.get("UNAME")
if UNAME is None:
    UNAME = NAME + "/up.txt"

plist = []


def write_fxn(d: dict[str, str]) -> bool:
    """Update requirements.txt."""
    if not d:
        return False

    with open(PNAME, encoding="utf-8") as fr:
        lines = fr.readlines()
        for i, li in enumerate(lines):
            line = li.strip(" \n")
            if not line:
                continue
            index = line.index("==")
            key = line[0:index]
            if key in d:
                lines[i] = line.replace(line[index + 2 :], d[key] + "\n")

    with open(PNAME, "w", encoding="utf-8") as fw:
        fw.writelines(lines)

    return True


def check(lines: list[str]) -> None:
    """Check if there are any update."""
    fd = {}
    for line in lines:
        wl = line.split()
        if wl[0] in plist:
            fd[wl[0]] = wl[2]

    rt = write_fxn(fd)

    if not rt:
        print("PIP_CHANGES=False")
    else:
        print("PIP_CHANGES=True")


def get_plist() -> None:
    """Get the package list from requirements.txt."""
    with open(PNAME, encoding="utf-8") as f:
        lines = f.readlines()
        for li in lines:
            line = li.strip(" \n")
            if not line:
                continue
            index = line.index("==")
            key = line[0:index]
            plist.append(key)


def main() -> None:
    """Run it to update new pip package versions in requirements.txt."""
    get_plist()
    with open(UNAME, encoding="utf-8") as f:
        lines = f.readlines()
        check(lines)


if __name__ == "__main__":
    main()
