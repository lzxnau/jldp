"""
Pip Package Upgrade Python Script.

JLDP Project.
:Author:  JLDP
:Date:    20231119
:Version: 1.0.2
"""


PNAME = "requirements.txt"
UNAME = "venv/up.txt"
plist = []


def write_fxn(d: dict) -> bool:
    "Update requirements.txt."
    if not d:
        return False

    with open(PNAME, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            line = line.strip(" \n")
            if not line:
                continue
            index = line.index("==")
            key = line[0:index]
            if key in d:
                lines[i] = line.replace(line[index + 2 :], d[key] + "\n")

    with open(PNAME, "w", encoding="utf-8") as fw:
        fw.writelines(lines)

    return True


def check(lines: list) -> None:
    "Check if there are any update."
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
    "Get the package list from requirements.txt."
    with open(PNAME, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip(" \n")
            if not line:
                continue
            index = line.index("==")
            key = line[0:index]
            plist.append(key)


def main() -> None:
    "Main fxn for the pip package upgrade."
    get_plist()
    with open(UNAME, "r", encoding="utf-8") as f:
        lines = f.readlines()
        check(lines)


if __name__ == "__main__":
    main()
