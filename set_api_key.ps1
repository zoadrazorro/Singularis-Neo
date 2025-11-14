# PowerShell script to set OpenAI API key for current session

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "OPENAI API KEY SETUP" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if already set
if ($env:OPENAI_API_KEY) {
    Write-Host "✓ API key already set: $($env:OPENAI_API_KEY.Substring(0, 7))..." -ForegroundColor Green
    $response = Read-Host "Do you want to change it? (y/n)"
    if ($response -ne "y") {
        Write-Host "Keeping existing API key" -ForegroundColor Yellow
        exit 0
    }
}

Write-Host "Enter your OpenAI API key (starts with sk-):" -ForegroundColor Yellow
$apiKey = Read-Host

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    Write-Host "ERROR: No API key entered" -ForegroundColor Red
    exit 1
}

if (-not $apiKey.StartsWith("sk-")) {
    Write-Host "WARNING: API key should start with 'sk-'" -ForegroundColor Yellow
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y") {
        exit 1
    }
}

# Set for current session
$env:OPENAI_API_KEY = $apiKey

Write-Host ""
Write-Host "✓ API key set for current PowerShell session" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run:" -ForegroundColor Cyan
Write-Host "  python run_beta_v3.py" -ForegroundColor White
Write-Host ""
Write-Host "Or test it:" -ForegroundColor Cyan
Write-Host "  python run_beta_v3.py --test-mode --duration 30" -ForegroundColor White
Write-Host ""
Write-Host "To make this permanent, add to your PowerShell profile:" -ForegroundColor Yellow
Write-Host "  `$env:OPENAI_API_KEY = '$apiKey'" -ForegroundColor White
Write-Host ""
