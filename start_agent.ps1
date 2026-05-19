$proc = Start-Process `
  -FilePath "E:\AI websocket\.venv313\Scripts\python.exe" `
  -ArgumentList @("agent.py", "dev") `
  -WorkingDirectory "E:\AI websocket\ebh-academy-ai" `
  -RedirectStandardOutput "E:\AI websocket\ebh-academy-ai\agent.log" `
  -RedirectStandardError "E:\AI websocket\ebh-academy-ai\agent_err.log" `
  -WindowStyle Hidden `
  -PassThru
Write-Host "Academy Agent PID: $($proc.Id)"
