import asyncio
from agents.razonamiento_planificacion.tools import FileReadTool, FileWriteTool, SystemBashTool
import os

async def main():
    print('--- Testing FileReadTool ---')
    reader = FileReadTool()
    with open('dummy.txt', 'w') as f:
        for i in range(1, 600):
            f.write(f'Line {i}\n')
    
    res1 = await reader.run('dummy.txt')
    print('Truncation detection:', '[TRUNCATED' in res1)
    
    res2 = await reader.run('{"path": "dummy.txt", "start_line": 300, "end_line": 302}')
    print('JSON pagination:', '300: Line 300' in res2)
    
    print('\n--- Testing FileWriteTool ---')
    writer = FileWriteTool()
    res_err = await writer.run('{"path": "dummy.txt", "old_string": "Lina 1\\n", "new_string": "Line One\\n"}')
    print('Difflib error detection:', 'Line 1' in res_err)
    
    print('\n--- Testing SystemBashTool ---')
    bash = SystemBashTool()
    res_bash = await bash.run('{"cmd": "sleep 2", "timeout": 1}')
    print('Dynamic timeout detection:', 'exceeded execution timeout of 1.0s' in res_bash)

    os.remove('dummy.txt')

asyncio.run(main())
