#Requires -Version 7
<#
.SYNOPSIS
    Управление системой: запуск, остановка, секреты, резервные копии, автозапуск.

.DESCRIPTION
    tm start | stop [-All] | restart | status | logs <сервис> | load-secrets
    tm autostart on|off
    tm secrets set <имя> [-FromFile путь] | edit <файл.yaml> | list | generate-key | export-recovery
    tm telegram-login
    tm backup | verify <файл> | restore <файл>
    tm eval seal|open
    tm watchdog

    Секреты хранятся зашифрованными DPAPI: расшифровать их можно только под этой учётной записью
    Windows. Контейнеры получают их через тома в памяти, поэтому на диске в открытом виде их нет.
#>
param(
    [Parameter(Position = 0)][string]$Command = "status",
    [Parameter(Position = 1)][string]$Argument,
    [Parameter(Position = 2)][string]$Value,
    [switch]$All,
    [string]$FromFile
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Security

$Root = Split-Path -Parent $PSScriptRoot
$Home_ = if ($env:TM_HOME) { $env:TM_HOME } else { Join-Path $env:LOCALAPPDATA "task-manager" }
$Store = Join-Path $Home_ "secrets"
$StoppedMark = Join-Path $Home_ "stopped"
$WatchState = Join-Path $Home_ "watchdog.json"
$Map = Get-Content (Join-Path $PSScriptRoot "secret-map.json") -Raw | ConvertFrom-Json
$BackupDir = if ($env:TM_BACKUP_DIR) { $env:TM_BACKUP_DIR } else { Join-Path $Root "backups" }
$Docker = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"
$DockerApp = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\Docker Desktop.exe"
$OllamaApp = if ($env:TM_OLLAMA_APP) { $env:TM_OLLAMA_APP } else {
    Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
}
$RunKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$RunName = "TaskManager"
$WatchTask = "TaskManagerWatchdog"
$BackupTask = "TaskManagerBackup"
$Services = @("collector", "anonymizer", "analyzer", "review_bot", "jira_export")
$MinFreeGb = 5

New-Item -ItemType Directory -Force $Store | Out-Null

function Fail([string]$Message) {
    [Console]::Error.WriteLine("Ошибка: $Message")
    exit 1
}

function Compose {
    # Callers read $LASTEXITCODE. Returning it too would mix it into captured output.
    Push-Location $Root
    try { & $Docker compose @args } finally { Pop-Location }
}

function Write-ToKeeper([string]$Folder, [string]$Name, [string]$Text) {
    # Through stdin, so the value never appears in a command line or in docker inspect.
    Push-Location $Root
    try {
        $Text | & $Docker compose exec -T keeper sh -c "umask 077; cat > /secrets/$Folder/$Name"
        if ($LASTEXITCODE -ne 0) { Fail "не удалось передать секрет $Name" }
    } finally { Pop-Location }
}

function Get-AllSecretNames {
    $names = @($Map.shared)
    foreach ($volume in $Map.volumes.PSObject.Properties) { $names += $volume.Value }
    return $names
}

function Test-KnownSecret([string]$Name) {
    if ($Name -notin (Get-AllSecretNames)) {
        Fail "неизвестный секрет '$Name'. Известные: $((Get-AllSecretNames) -join ', ')"
    }
}

function Write-Secret([string]$Name, [byte[]]$Bytes) {
    $protected = [System.Security.Cryptography.ProtectedData]::Protect(
        $Bytes, $null, [System.Security.Cryptography.DataProtectionScope]::CurrentUser)
    [System.IO.File]::WriteAllBytes((Join-Path $Store "$Name.dpapi"), $protected)
}

function Read-Secret([string]$Name) {
    $path = Join-Path $Store "$Name.dpapi"
    if (-not (Test-Path $path)) { return $null }
    return [System.Security.Cryptography.ProtectedData]::Unprotect(
        [System.IO.File]::ReadAllBytes($path), $null,
        [System.Security.Cryptography.DataProtectionScope]::CurrentUser)
}

function Read-SecretText([string]$Name) {
    $bytes = Read-Secret $Name
    if ($null -eq $bytes) { return $null }
    return [System.Text.Encoding]::UTF8.GetString($bytes)
}

function Wait-Docker {
    & $Docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -eq 0) { return }
    if (Test-Path $DockerApp) { Start-Process $DockerApp }
    $deadline = (Get-Date).AddMinutes(5)
    do {
        Start-Sleep 5
        & $Docker info --format "{{.ServerVersion}}" *> $null
        if ($LASTEXITCODE -eq 0) { return }
    } until ((Get-Date) -gt $deadline)
    Fail "Docker не запустился за 5 минут"
}

function Start-Ollama {
    try {
        Invoke-RestMethod "http://127.0.0.1:11434/api/version" -TimeoutSec 3 | Out-Null
    } catch {
        if (Test-Path $OllamaApp) { Start-Process $OllamaApp }
    }
}

function Push-Secrets {
    $missing = Get-AllSecretNames | Where-Object { -not (Test-Path (Join-Path $Store "$_.dpapi")) }
    if ($missing) {
        Fail "не заданы секреты: $($missing -join ', '). Задайте их: tm secrets set <имя>"
    }
    foreach ($volume in $Map.volumes.PSObject.Properties) {
        foreach ($name in (@($Map.shared) + @($volume.Value))) {
            Write-ToKeeper $volume.Name $name (Read-SecretText $name)
        }
    }
}

function Invoke-Start {
    Remove-Item $StoppedMark -Force -ErrorAction SilentlyContinue
    Wait-Docker
    Start-Ollama
    Compose up -d --build keeper | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "не удалось запустить keeper" }
    Push-Secrets
    Compose up -d --wait @Services | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "не все сервисы запустились: tm status, tm logs <сервис>" }
    Write-Host "Система запущена."
}

function Invoke-Stop {
    # Marked first, so the watchdog stays quiet and nothing brings the system back by itself.
    New-Item -ItemType File -Force $StoppedMark | Out-Null
    & $Docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -eq 0) {
        Compose stop | Out-Null
    }
    if ($All) {
        Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue |
            Stop-Process -Force -Confirm:$false
        Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue |
            Stop-Process -Force -Confirm:$false
        Write-Host "Система, Ollama и Docker Desktop остановлены."
        return
    }
    Write-Host "Система остановлена. Сообщения за это время будут дочитаны при запуске."
}

function Get-FreeGb([string]$Letter) {
    return [math]::Round((Get-Volume -DriveLetter $Letter).SizeRemaining / 1GB)
}

function Get-DriveLetters([string[]]$Paths) {
    return @($Paths | Where-Object { $_ -match "^[A-Za-z]:" } |
        ForEach-Object { $_.Substring(0, 1).ToUpper() } | Select-Object -Unique)
}

function Get-LatestBackup {
    if (-not (Test-Path $BackupDir)) { return $null }
    return Get-ChildItem $BackupDir -Filter "app-*.db" -ErrorAction SilentlyContinue |
        Sort-Object Name | Select-Object -Last 1
}

function Invoke-Status {
    Write-Host "Автозапуск: $(if (Get-ItemProperty $RunKey -Name $RunName -ErrorAction SilentlyContinue) { 'вкл' } else { 'выкл' })"
    Write-Host "Остановлена намеренно: $(if (Test-Path $StoppedMark) { 'да' } else { 'нет' })"
    & $Docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) { Write-Host "Docker: не работает" } else {
        Compose ps --format "table {{.Service}}\t{{.Status}}"
    }
    try {
        Invoke-RestMethod "http://127.0.0.1:11434/api/version" -TimeoutSec 3 | Out-Null
        Write-Host "Ollama: работает"
    } catch { Write-Host "Ollama: не работает" }
    $backup = Get-LatestBackup
    Write-Host "Последняя резервная копия: $(if ($backup) { $backup.Name } else { 'нет' })"
    foreach ($letter in (Get-DriveLetters $env:SystemDrive, $Root, $BackupDir)) { Write-Host "Свободно на ${letter}: $(Get-FreeGb $letter) ГБ" }
}

function Invoke-Secrets {
    switch ($Argument) {
        "set" {
            if (-not $Value) { Fail "укажите имя: tm secrets set <имя>" }
            Test-KnownSecret $Value
            if ($FromFile) {
                Write-Secret $Value ([System.IO.File]::ReadAllBytes($FromFile))
            } else {
                # Hidden input: the value appears neither on screen nor in the shell's history.
                $secure = Read-Host "Значение $Value" -AsSecureString
                $plain = [System.Net.NetworkCredential]::new("", $secure).Password
                Write-Secret $Value ([System.Text.Encoding]::UTF8.GetBytes($plain))
            }
            Write-Host "Сохранено: $Value. Применится при следующем tm start."
        }
        "edit" {
            if (-not $Value) { Fail "укажите файл: tm secrets edit me.yaml" }
            Test-KnownSecret $Value
            if ($Value -notmatch "\.yaml$") { Fail "редактируются только файлы .yaml" }
            $temp = Join-Path $Home_ "edit-$Value"
            $original = Read-Secret $Value
            if ($null -ne $original) { [IO.File]::WriteAllBytes($temp, $original) }
            try {
                Start-Process notepad $temp -Wait
                Write-Secret $Value ([IO.File]::ReadAllBytes($temp))
            } finally {
                # Overwritten first: the file held the plain text for as long as it was open.
                if (Test-Path $temp) {
                    $size = [math]::Max((Get-Item $temp).Length, 1)
                    [IO.File]::WriteAllBytes($temp, [byte[]]::new($size))
                    Remove-Item $temp -Force
                }
            }
            Write-Host "Сохранено: $Value. Применить: tm restart"
        }
        "list" {
            foreach ($name in Get-AllSecretNames) {
                $state = if (Test-Path (Join-Path $Store "$name.dpapi")) { "задан" } else { "НЕ ЗАДАН" }
                Write-Host "$name`: $state"
            }
        }
        "generate-key" {
            if (Test-Path (Join-Path $Store "database_key.dpapi")) {
                Fail "ключ базы уже есть. Новый ключ сделал бы базу и все копии нечитаемыми."
            }
            $bytes = [System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
            $hex = -join ($bytes | ForEach-Object { $_.ToString("x2") })
            Write-Secret "database_key" ([System.Text.Encoding]::UTF8.GetBytes($hex))
            Write-Host "Ключ базы создан. Сохраните копию: tm secrets export-recovery"
        }
        "export-recovery" {
            $key = Read-SecretText "database_key"
            if (-not $key) { Fail "ключа базы ещё нет" }
            Write-Host "Без этого ключа после переустановки Windows база и копии будут нечитаемы."
            Write-Host "Сохраните его в менеджере паролей и очистите экран. В чат его не отправляйте."
            Write-Host ""
            Write-Host $key
        }
        default { Fail "tm secrets set <имя> | edit <файл.yaml> | list | generate-key | export-recovery" }
    }
}

function Invoke-TelegramLogin {
    Wait-Docker
    Compose up -d --build keeper | Out-Null
    foreach ($name in "tg_api_id", "tg_api_hash", "database_key") {
        $text = Read-SecretText $name
        if (-not $text) { Fail "сначала задайте секрет $name" }
        Write-ToKeeper "collector" $name $text
    }
    # The session lands in the in-memory volume and moves straight to the protected store.
    Compose run --rm --no-deps -v task-manager_secrets_collector:/session_out -it collector `
        python -m task_manager login --save-to /session_out/tg_session
    if ($LASTEXITCODE -ne 0) { Fail "вход не выполнен" }
    $session = Compose exec -T keeper sh -c "cat /secrets/collector/tg_session"
    if (-not $session) { Fail "сессия не получена" }
    Write-Secret "tg_session" ([System.Text.Encoding]::UTF8.GetBytes(($session | Out-String).Trim()))
    Write-Host "Сессия Telegram сохранена в защищённом хранилище."
}

function Send-ToOwner([string]$Text) {
    # The bot can't report its own failure, so this goes straight to Telegram.
    $token = Read-SecretText "bot_token"
    $owner = Read-SecretText "bot_owner_id"
    if (-not $token -or -not $owner) { return }
    try {
        Invoke-RestMethod -Method Post "https://api.telegram.org/bot$($token.Trim())/sendMessage" `
            -Body @{ chat_id = $owner.Trim(); text = $Text } -TimeoutSec 20 | Out-Null
    } catch {
        # No connection: there's nobody to tell, and the next run tries again.
        # The exception's own message could hold the URL, and with it the token.
        throw "send failed: $($_.Exception.GetType().Name)"
    }
}

function Get-Problems {
    $problems = [ordered]@{}
    & $Docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) {
        $problems["docker"] = "Docker не работает, система остановлена"
    } else {
        Push-Location $Root
        try {
            # A stopped service gives no output, and comparing an empty result gives nothing truthy.
            $bot = "$(& $Docker compose ps review_bot --format "{{.Status}}")"
        } finally { Pop-Location }
        if ($bot -notmatch "\(healthy\)") {
            $problems["bot"] = "Бот не работает: сообщения о задачах не приходят"
        }
    }
    foreach ($letter in (Get-DriveLetters $env:SystemDrive, $Root)) {
        if ((Get-FreeGb $letter) -lt $MinFreeGb) {
            $problems["disk_$letter"] = "На диске $letter`: меньше $MinFreeGb ГБ свободно"
        }
    }
    $backup = Get-LatestBackup
    if ($backup -and ((Get-Date) - $backup.LastWriteTime).TotalDays -gt 3) {
        $problems["backup"] = "Резервной копии не было больше 3 дней"
    }
    return $problems
}

function Invoke-Watchdog {
    # A deliberate stop raises no alarm.
    if (Test-Path $StoppedMark) { return }
    $known = @{}
    if (Test-Path $WatchState) {
        (Get-Content $WatchState -Raw | ConvertFrom-Json).PSObject.Properties |
            ForEach-Object { $known[$_.Name] = $_.Value }
    }
    $problems = Get-Problems
    try {
        foreach ($name in $problems.Keys) {
            if (-not $known.ContainsKey($name)) {
                Send-ToOwner "⚠️ $($problems[$name])"
                $known[$name] = $problems[$name]
            }
        }
        foreach ($name in @($known.Keys)) {
            if (-not $problems.Contains($name)) {
                Send-ToOwner "✅ Исправлено: $($known[$name])"
                $known.Remove($name)
            }
        }
    } catch {
        # Told next time. What was already sent stays recorded below, and this leaves a trace,
        # because a watchdog failing silently is exactly what it exists to prevent.
        "$(Get-Date -Format s) $($_.Exception.GetType().Name)" |
            Add-Content (Join-Path $Home_ "watchdog.log")
    }
    $known | ConvertTo-Json | Set-Content $WatchState -Encoding utf8
}

function Invoke-Autostart {
    switch ($Argument) {
        "on" {
            $script = Join-Path $PSScriptRoot "tm.ps1"
            $pwsh = (Get-Command pwsh).Source
            # Windows opens a console before pwsh can hide it, which flashes on screen every run.
            # A headless conhost gives pwsh a console that is never shown.
            $conhost = Join-Path $env:WINDIR "System32\conhost.exe"
            $hidden = "--headless `"$pwsh`" -NoProfile -File `"$script`""
            Set-ItemProperty $RunKey -Name $RunName -Value "`"$conhost`" $hidden autostart-run"
            $watch = New-ScheduledTaskAction -Execute $conhost -Argument "$hidden watchdog"
            $every = New-ScheduledTaskTrigger -Once -At (Get-Date) `
                -RepetitionInterval (New-TimeSpan -Minutes 5)
            Register-ScheduledTask -Force -TaskName $WatchTask -Action $watch -Trigger $every | Out-Null
            $save = New-ScheduledTaskAction -Execute $conhost -Argument "$hidden backup"
            $daily = New-ScheduledTaskTrigger -Daily -At "13:00"
            $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
            Register-ScheduledTask -Force -TaskName $BackupTask -Action $save -Trigger $daily `
                -Settings $settings | Out-Null
            Write-Host "Автозапуск включён: при входе в Windows, проверка каждые 5 минут, копия раз в день."
        }
        "off" {
            Remove-ItemProperty $RunKey -Name $RunName -ErrorAction SilentlyContinue
            foreach ($task in $WatchTask, $BackupTask) {
                Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue
            }
            Write-Host "Автозапуск выключен."
        }
        default { Fail "tm autostart on|off" }
    }
}

function Invoke-Backup {
    Wait-Docker
    New-Item -ItemType Directory -Force $BackupDir | Out-Null
    $env:TM_BACKUP_DIR = $BackupDir
    Compose run --rm --no-deps backup
    if ($LASTEXITCODE -ne 0) {
        try { Send-ToOwner "⚠️ Резервная копия не создана. Проверьте: tm backup" } catch {}
        Fail "резервная копия не создана"
    }
}

function Invoke-Verify([string]$Name) {
    $env:TM_BACKUP_DIR = $BackupDir
    Compose run --rm --no-deps backup python -m task_manager verify-backup "/backups/$Name"
    if ($LASTEXITCODE -ne 0) { Fail "копия не читается" }
}

function Invoke-Restore([string]$Name) {
    if (-not $Name) { Fail "tm restore <файл из $BackupDir>" }
    Invoke-Verify $Name
    $answer = Read-Host "Текущая база будет заменена копией $Name. Напишите ДА, чтобы продолжить"
    if ($answer -cne "ДА") { Fail "отменено" }
    Compose stop @Services | Out-Null
    $env:TM_BACKUP_DIR = $BackupDir
    Compose run --rm --no-deps backup python -m task_manager restore "/backups/$Name"
    if ($LASTEXITCODE -ne 0) { Fail "не удалось восстановить" }
    Invoke-Start
}

function Invoke-Eval {
    if ($Argument -notin "seal", "open") { Fail "tm eval seal|open" }
    $key = Read-SecretText "database_key"
    if (-not $key) { Fail "ключа базы ещё нет" }
    Push-Location $Root
    $env:DATABASE_KEY = $key.Trim()
    try {
        & uv run python eval/archive.py $Argument
        if ($LASTEXITCODE -ne 0) { Fail "архив не обработан" }
    } finally {
        Remove-Item Env:DATABASE_KEY -ErrorAction SilentlyContinue
        Pop-Location
    }
}

switch ($Command) {
    "start" { Invoke-Start }
    "stop" { Invoke-Stop }
    "restart" { Invoke-Stop; Invoke-Start }
    "status" { Invoke-Status }
    "logs" {
        if ($Argument -notin $Services) { Fail "сервисы: $($Services -join ', ')" }
        Compose logs --no-log-prefix --tail 100 $Argument
    }
    "secrets" { Invoke-Secrets }
    "telegram-login" { Invoke-TelegramLogin }
    "load-secrets" {
        Wait-Docker
        Compose up -d --build keeper | Out-Null
        Push-Secrets
        Write-Host "Секреты загружены в память."
    }
    "autostart" { Invoke-Autostart }
    "autostart-run" {
        # A deliberate stop holds across reboots, until tm start.
        if (-not (Test-Path $StoppedMark)) { Invoke-Start }
    }
    "watchdog" { Invoke-Watchdog }
    "backup" { Invoke-Backup }
    "verify" { Invoke-Verify $Argument }
    "restore" { Invoke-Restore $Argument }
    "eval" { Invoke-Eval }
    default { Fail "неизвестная команда '$Command'. Подробности: Get-Help scripts\tm.ps1" }
}
