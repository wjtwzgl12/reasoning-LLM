# PHYRE paper1 — git init + first commit + push
#
# Usage (PowerShell, in project root):
#   # 1. 在 GitHub 创建私有 repo phyre-paper1(空仓,不要 README/license)
#   # 2. 编辑下面 REMOTE_URL 为你的仓库地址
#   # 3. 运行:  .\init_git.ps1
#
# 该脚本会:
#   - git init(如未初始化)
#   - 配置 core.autocrlf / core.quotepath(支持中文路径)
#   - 挑选要追踪的文件(白名单 + .gitignore 黑名单)
#   - 创建首个 commit
#   - 添加 remote + push -u

param(
    [string]$RemoteUrl = "git@github.com:<YOUR_USER>/phyre-paper1.git",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

Write-Host "== PHYRE git init ==" -ForegroundColor Cyan
Write-Host "remote: $RemoteUrl"
Write-Host "branch: $Branch"
Write-Host ""

if ($RemoteUrl -like "*<YOUR_USER>*") {
    Write-Host "ERROR: 请先编辑脚本里的 RemoteUrl 或以参数传入:" -ForegroundColor Red
    Write-Host "  .\init_git.ps1 -RemoteUrl git@github.com:wjt/phyre-paper1.git" -ForegroundColor Yellow
    exit 1
}

# 1. init
if (-not (Test-Path ".git")) {
    git init -b $Branch
} else {
    Write-Host "(已存在 .git,跳过 init)" -ForegroundColor Yellow
}

# 2. config(支持 Windows 中文路径 + LF 行尾)
git config core.quotepath false
git config core.autocrlf input
git config core.safecrlf false

# 3. 白名单添加(.gitignore 已经过滤掉大文件/recovery 脚本)
#    目录级 add,配合 .gitignore 黑名单即可
Write-Host "`n== staging files ==" -ForegroundColor Cyan
git add .gitignore
git add init_git.ps1
git add "Paper1_架构设计_L1-L4.md"
git add "Paper1_实验路线_v1.md"
git add "Paper1_骨架.md"

# 源码与脚本(递归 add,过滤交给 .gitignore)
if (Test-Path "pvgap_experiment/src")        { git add "pvgap_experiment/src" }
if (Test-Path "pvgap_experiment/scripts")    { git add "pvgap_experiment/scripts" }
if (Test-Path "pvgap_experiment/prompts")    { git add "pvgap_experiment/prompts" }
if (Test-Path "pvgap_experiment/colab")      { git add "pvgap_experiment/colab" }
if (Test-Path "nb")                          { git add "nb" }

# 其他关键 md(记录性文档)
Get-ChildItem -Filter "*.md" -File | ForEach-Object { git add $_.Name }

# 4. 状态确认
Write-Host "`n== git status (staged) ==" -ForegroundColor Cyan
git status --short

# 5. 首次 commit
$commitMsg = @"
init: PHYRE paper1 baseline

- Paper1_架构设计_L1-L4.md v1 (4-layer architecture: ontology / MCTS+PRM /
  MI-selection / BOED)
- Paper1_实验路线_v1.md v1 (方案A 严格对齐架构,8-12 周 Colab 并行路线)
- Paper1_骨架.md (v0.5, 含 §9E.1 empirical lock)
- pvgap_experiment src/scripts/prompts/colab snapshot from §9E.1 run
- .gitignore 过滤 ckpt/zip/_RECOVERED*/scratch
"@

git commit -m $commitMsg

# 6. remote + push
$existingRemote = git remote 2>$null
if ($existingRemote -contains "origin") {
    Write-Host "(origin 已存在,更新 URL)" -ForegroundColor Yellow
    git remote set-url origin $RemoteUrl
} else {
    git remote add origin $RemoteUrl
}

Write-Host "`n== push to origin ==" -ForegroundColor Cyan
git push -u origin $Branch

Write-Host "`n✓ 完成。下次增量提交:  git add <files>  &&  git commit -m '...'  &&  git push" -ForegroundColor Green
