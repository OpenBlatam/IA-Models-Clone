# PowerShell script to rename files and directories to lowercase with underscores
# This script will recursively process the instagram_captions directory

param(
    [string]$Path = "."
)

function Convert-ToSnakeCase {
    param([string]$Name)
    
    # Remove file extension first
    $nameWithoutExt = $Name
    $extension = ""
    if ($Name -match "\.(.+)$") {
        $nameWithoutExt = $Name -replace "\.(.+)$", ""
        $extension = $matches[1]
    }
    
    # Convert to snake_case
    # Handle camelCase and PascalCase
    $snakeCase = $nameWithoutExt -replace "([a-z0-9])([A-Z])", '$1_$2'
    # Handle consecutive uppercase letters
    $snakeCase = $snakeCase -replace "([A-Z])([A-Z][a-z])", '$1_$2'
    # Convert to lowercase and replace spaces/hyphens with underscores
    $snakeCase = $snakeCase.ToLower() -replace "[- ]+", "_"
    # Remove any remaining non-alphanumeric characters except underscores
    $snakeCase = $snakeCase -replace "[^a-z0-9_]", "_"
    # Remove multiple consecutive underscores
    $snakeCase = $snakeCase -replace "_+", "_"
    # Remove leading/trailing underscores
    $snakeCase = $snakeCase.Trim("_")
    
    # Add extension back if it existed
    if ($extension) {
        $snakeCase = "$snakeCase.$($extension.ToLower())"
    }
    
    return $snakeCase
}

function Should-SkipDirectory {
    param([string]$DirName)
    
    $skipDirs = @("__pycache__", ".git", ".vscode", ".idea", "node_modules", "venv", "env", ".env", "dist", "build", "target")
    return $skipDirs -contains $DirName
}

function Rename-FilesAndDirectories {
    param([string]$RootPath)
    
    $rootPathObj = Get-Item $RootPath
    
    # Get all files and directories recursively
    $allItems = Get-ChildItem -Path $RootPath -Recurse | Sort-Object FullName -Descending
    
    $renamedCount = 0
    
    foreach ($item in $allItems) {
        # Skip certain directories
        if ($item.PSIsContainer -and (Should-SkipDirectory $item.Name)) {
            continue
        }
        
        $newName = Convert-ToSnakeCase $item.Name
        
        if ($newName -ne $item.Name) {
            try {
                $newPath = Join-Path $item.Directory.FullName $newName
                Rename-Item -Path $item.FullName -NewName $newName -Force
                Write-Host "Renamed: $($item.FullName) -> $newPath"
                $renamedCount++
            }
            catch {
                Write-Host "Error renaming $($item.FullName): $_"
            }
        }
    }
    
    Write-Host "`nTotal files/directories renamed: $renamedCount"
}

# Main execution
Write-Host "Starting file and directory renaming process..."
Write-Host "Working directory: $(Get-Location)"
Write-Host "Converting all names to lowercase with underscores..."
Write-Host ""

# Auto-confirm for automation
$response = "y"
if ($response -ne "y") {
    Write-Host "Operation cancelled."
    exit
}

try {
    Rename-FilesAndDirectories -RootPath $Path
    Write-Host "`nRenaming process completed successfully!"
    Write-Host "`nNote: You may need to update import statements in your code files."
    Write-Host "Consider running a search and replace for old import paths."
}
catch {
    Write-Host "Error during renaming process: $_"
} 