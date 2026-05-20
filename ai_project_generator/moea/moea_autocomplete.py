"""
MOEA Autocomplete - Autocompletado para shells
=============================================
Genera scripts de autocompletado para bash, zsh y PowerShell
"""
from pathlib import Path


def generate_bash_completion():
    """Generar autocompletado para bash"""
    completion = """# MOEA CLI Bash Completion
_moea_cli_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="generate setup verify test benchmark visualize status monitor export health utils config"

    if [[ ${cur} == -* ]] ; then
        COMPREPLY=( $(compgen -W "-h --help -v --verbose" -- ${cur}) )
        return 0
    fi

    case "${prev}" in
        generate)
            COMPREPLY=( $(compgen -W "--name --author --verbose" -- ${cur}) )
            ;;
        setup)
            COMPREPLY=( $(compgen -W "--project-dir --no-backend --no-frontend" -- ${cur}) )
            ;;
        test|benchmark|monitor|health)
            COMPREPLY=( $(compgen -W "--url" -- ${cur}) )
            ;;
        visualize|export)
            COMPREPLY=( $(compgen -W "--output --url" -- ${cur}) )
            ;;
        *)
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            ;;
    esac
}

complete -F _moea_cli_completion moea_cli.py
complete -F _moea_cli_completion python moea_cli.py
"""
    return completion


def generate_zsh_completion():
    """Generar autocompletado para zsh"""
    completion = """#compdef moea_cli.py

_moea_cli() {
    local context state line
    typeset -A opt_args

    _arguments \\
        '1: :->command' \\
        '*:: :->args'

    case $state in
        command)
            _values "commands" \\
                "generate[Generate MOEA project]" \\
                "setup[Setup project]" \\
                "verify[Verify project]" \\
                "test[Test API]" \\
                "benchmark[Run benchmarks]" \\
                "visualize[Visualize results]" \\
                "status[Check status]" \\
                "monitor[Monitor system]" \\
                "export[Export project]" \\
                "health[Health check]" \\
                "utils[Utilities]" \\
                "config[Configuration]"
            ;;
        args)
            case $words[1] in
                generate)
                    _arguments \\
                        '--name[Project name]' \\
                        '--author[Author name]' \\
                        '--verbose[Verbose output]'
                    ;;
                setup)
                    _arguments \\
                        '--project-dir[Project directory]' \\
                        '--no-backend[Skip backend]' \\
                        '--no-frontend[Skip frontend]'
                    ;;
            esac
            ;;
    esac
}

_moea_cli "$@"
"""
    return completion


def generate_powershell_completion():
    """Generar autocompletado para PowerShell"""
    completion = """# MOEA CLI PowerShell Completion
Register-ArgumentCompleter -Native -CommandName moea_cli.py -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    $commands = @(
        'generate',
        'setup',
        'verify',
        'test',
        'benchmark',
        'visualize',
        'status',
        'monitor',
        'export',
        'health',
        'utils',
        'config'
    )

    $commands | Where-Object {
        $_ -like "$wordToComplete*"
    } | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}
"""
    return completion


def install_completion(shell: str = "bash"):
    """Instalar autocompletado"""
    completions = {
        "bash": generate_bash_completion(),
        "zsh": generate_zsh_completion(),
        "powershell": generate_powershell_completion()
    }
    
    if shell not in completions:
        print(f"❌ Shell no soportado: {shell}")
        print(f"   Shells disponibles: {', '.join(completions.keys())}")
        return False
    
    completion = completions[shell]
    output_file = f"moea_completion.{shell}"
    
    with open(output_file, 'w') as f:
        f.write(completion)
    
    print(f"✅ Autocompletado generado: {output_file}")
    print(f"\nPara instalar:")
    
    if shell == "bash":
        print(f"   echo 'source {Path(output_file).absolute()}' >> ~/.bashrc")
        print(f"   source ~/.bashrc")
    elif shell == "zsh":
        print(f"   echo 'source {Path(output_file).absolute()}' >> ~/.zshrc")
        print(f"   source ~/.zshrc")
    elif shell == "powershell":
        print(f"   . {Path(output_file).absolute()}")
        print(f"   # O agregar al perfil: Add-Content $PROFILE (Get-Content {output_file})")
    
    return True


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Autocomplete Generator")
    parser.add_argument(
        '--shell',
        choices=['bash', 'zsh', 'powershell'],
        default='bash',
        help='Shell para generar autocompletado'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generar para todos los shells'
    )
    
    args = parser.parse_args()
    
    if args.all:
        for shell in ['bash', 'zsh', 'powershell']:
            install_completion(shell)
            print()
    else:
        install_completion(args.shell)


if __name__ == "__main__":
    main()

