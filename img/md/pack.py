apps = 'pip setuptools wheel'

# Sphinx Packages
apps += ' Sphinx myst-parser sphinx_rtd_theme sphinx_gallery'
apps += ' sphinx_design sphinxcontrib-mermaid nbsphinx ipykernel'
apps += ' sphinx-copybutton sphinx-togglebutton'

# Develop Packages
apps += ' numpy pandas matplotlib seaborn scipy scikit-learn'
apps += ' opencv-python albumentations'
apps += ' torch torchtext torchaudio torchvision'


def writeFx(d: dict) -> bool:
  if len(d) == 0:
    return False
  
  name = 'requirements.txt'
  
  with open(name, 'r') as fr:
    lines = fr.readlines()
    for i, line in enumerate(lines):
      line = line.strip(' \n')
      if len(line) == 0:
        continue
      index = line.index('==')
      key = line[0:index]
      if key in d:
        lines[i] = line.replace(line[index + 2:], d[key] + '\n')
  
  with open(name, 'w') as fw:
    fw.writelines(lines)
  
  return True


def check(lines: list) -> None:
  fd = dict()
  for line in lines:
    wl = line.split()
    if wl[0] in apps:
      fd[wl[0]] = wl[2]
  
  rt = writeFx(fd)
  
  if rt == 0:
    print('PIP_CHANGES=False')
  else:
    print('PIP_CHANGES=True')


def main() -> None:
  with open("./venv/up.txt", "r") as f:
    lines = f.readlines()
    check(lines)


if __name__ == '__main__':
  main()
