"""Script to wrap remaining imports in try-except blocks"""
import re

file_path = 'robot_movement_ai/core/__init__.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find all unwrapped imports (from .xxx import that don't have try: before them)
lines = content.split('\n')
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if this is an unwrapped import
    if line.strip().startswith('from .') and 'import' in line:
        # Check if previous lines have try: (within last 10 lines)
        has_try = False
        for j in range(max(0, i-10), i):
            if 'try:' in lines[j] and not any('except ImportError:' in lines[k] for k in range(j, i)):
                has_try = False
                break
            if 'except ImportError:' in lines[j]:
                has_try = True
                break
        
        if not has_try:
            # Find the end of this import block
            import_start = i
            import_lines = [line]
            i += 1
            indent_level = len(line) - len(line.lstrip())
            
            # Collect all lines until we find the closing parenthesis
            open_parens = line.count('(') - line.count(')')
            while i < len(lines) and (open_parens > 0 or lines[i].strip() == '' or lines[i].strip().startswith('#')):
                if lines[i].strip() == '':
                    import_lines.append(lines[i])
                    i += 1
                    continue
                if lines[i].strip().startswith('#'):
                    import_lines.append(lines[i])
                    i += 1
                    continue
                import_lines.append(lines[i])
                open_parens += lines[i].count('(') - lines[i].count(')')
                if open_parens == 0:
                    i += 1
                    break
                i += 1
            
            # Extract import names
            import_block = '\n'.join(import_lines)
            # Extract names from import
            names = re.findall(r'(\w+)(?:\s+as\s+\w+)?', import_block)
            names = [n for n in names if n not in ['from', 'import', 'as']]
            
            # Wrap in try-except
            new_lines.append('# ' + import_lines[0].replace('from .', 'from .'))
            new_lines.append('try:')
            for import_line in import_lines:
                if import_line.strip().startswith('#'):
                    new_lines.append('    ' + import_line)
                elif import_line.strip():
                    new_lines.append('    ' + import_line)
                else:
                    new_lines.append(import_line)
            
            new_lines.append('except ImportError:')
            for name in set(names):
                if name and name[0].isupper() or name.startswith('get_') or name.startswith('create_'):
                    new_lines.append(f'    {name} = None')
            continue
    
    new_lines.append(line)
    i += 1

with open(file_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

print("Fixed imports")



