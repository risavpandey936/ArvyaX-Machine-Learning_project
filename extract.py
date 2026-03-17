import json

with open('Untitled67.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open('extracted_code.py', 'w', encoding='utf-8') as out:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            out.write(''.join(cell['source']))
            out.write('\n\n# ---\n\n')
