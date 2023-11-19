"""
Pip Package Upgrade Python Script

:Author:  JLDP
:Date:    20231119
:Version: 1.0.1
"""

pname = 'requirements.txt'
uname = 'venv/up.txt'
plist = []

def writeFx(d: dict) -> bool:
  if len(d) == 0:
    return False

  with open(pname, 'r') as fr:
    lines = fr.readlines()
    for i, line in enumerate(lines):
      line = line.strip(' \n')
      if len(line) == 0:
        continue
      index = line.index('==')
      key = line[0:index]
      if key in d:
        lines[i] = line.replace(line[index + 2:], d[key] + '\n')

  with open(pname, 'w') as fw:
    fw.writelines(lines)

  return True

def check(lines: list) -> None:
  fd = dict()
  for line in lines:
    wl = line.split()
    if wl[0] in plist:
      fd[wl[0]] = wl[2]

  rt = writeFx(fd)

  if rt == 0:
    print('PIP_CHANGES=False')
  else:
    print('PIP_CHANGES=True')

def getPList() -> None:
  with open(pname, 'r') as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip(' \n')
      if len(line) == 0:
        continue
      index = line.index('==')
      key = line[0:index]
      plist.append(key)

def main() -> None:
  getPList()
  with open(uname, 'r') as f:
    lines = f.readlines()
    check(lines)


if __name__ == '__main__':
  main()
