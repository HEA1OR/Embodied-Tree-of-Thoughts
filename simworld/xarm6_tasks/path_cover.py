#!/usr/bin/env python3

FOLDER      = r'your_path/Embodied-Tree-of-Thoughts/simworld'       
OLD_STRING  = '/home/dell/workspace/xwj/Embodied-Tree-of-Thoughts/simworld'          
NEW_STRING  = 'your_path/Embodied-Tree-of-Thoughts/simworld'          

import os
import chardet
from pathlib import Path

def detect_encoding(path: Path) -> str:
    with path.open('rb') as f:
        return chardet.detect(f.read())['encoding'] or 'utf-8'

def replace_in_file(file_path: Path) -> bool:
    enc = detect_encoding(file_path)
    text = file_path.read_text(encoding=enc, errors='ignore')
    if OLD_STRING not in text:
        return False
    file_path.write_text(text.replace(OLD_STRING, NEW_STRING), encoding=enc)
    print(f'✅   {file_path}')
    return True

def main():
    root = Path(FOLDER).expanduser().resolve()
    if not root.is_dir():
        print(f'{root} not a valid directory'); return

    total = 0
    for ext in ('*.py', '*.json'):
        for file in root.rglob(ext):
            if replace_in_file(file):
                total += 1
    print(f'\n {total} files replaced')

if __name__ == '__main__':
    main()