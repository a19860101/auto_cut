# AI 自動剪輯工具

這是一個 Streamlit 網頁專案，可以上傳 `MP4`、`MOV`、`MKV` 影片，自動偵測有人講話的區段，剪除超過指定長度的無語音停頓，最後輸出成 `MP4`。

## 功能

- 上傳 MP4 / MOV / MKV
- 設定「最短停頓秒數」
- 設定語音前後 padding，避免切到尾音
- 使用 FFmpeg 抽取單聲道 16kHz WAV
- 使用 Silero VAD 偵測 speech timestamps
- 合併相鄰語音區間，只刪除大於指定門檻的無語音區段
- 使用 FFmpeg `trim` / `atrim` / `concat` 輸出 MP4
- 顯示原始長度、剪輯後長度、刪除秒數
- 提供下載按鈕

## 系統需求

- 建議 Python 3.10 或 3.11
- FFmpeg 與 FFprobe
- 1GB 以上記憶體
- 支援 PyTorch 的 CPU 環境

Silero VAD 官方套件支援 Python 3.8+，且模型本身需要 `torch>=1.12.0`。實務上，若遇到 PyTorch wheel 或 torch 相容性問題，建議優先使用 Python 3.10 或 3.11 建立虛擬環境。

## 安裝

### 1. 安裝 FFmpeg

macOS：

```bash
brew install ffmpeg
```

Ubuntu / Debian：

```bash
sudo apt update
sudo apt install ffmpeg
```

Windows：

建議從 FFmpeg 官方網站下載，或使用 Chocolatey：

```powershell
choco install ffmpeg
```

安裝後請確認：

```bash
ffmpeg -version
ffprobe -version
```

### 2. 建立虛擬環境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. 安裝 Python 套件

```bash
pip install -r requirements.txt
```

如果安裝 `torch` 失敗，請先依照 PyTorch 官方安裝頁面選擇適合你作業系統的 CPU 版本安裝指令，安裝完成後再執行：

```bash
pip install streamlit "silero-vad>=6.2,<7"
```

本專案刻意不使用 `silero-vad[audio]`，而是先用 FFmpeg 抽出 16kHz WAV 再交給 Silero VAD，這樣可以避開部分 `torchaudio` / `torchcodec` / FFmpeg 版本相容性問題。

## 執行

```bash
streamlit run app.py
```

開啟瀏覽器後，上傳影片並設定：

- 最短停頓秒數：例如 `0.8` 秒，代表只有超過 0.8 秒的無語音段落才會被剪掉
- 前後 padding：例如 `0.15` 秒，代表每段語音前後多保留 0.15 秒

按下「開始自動剪輯」後，完成時會顯示統計資訊並提供 MP4 下載。

## 上傳到 Streamlit Cloud

這個專案已經準備好部署到 Streamlit Cloud：

- `requirements.txt`：安裝 Python 套件
- `packages.txt`：讓 Streamlit Cloud 安裝 FFmpeg
- `.streamlit/config.toml`：提高上傳檔案大小限制
- `.gitignore`：避免把 `.venv`、`.idea`、系統快取上傳到 GitHub

### 1. 上傳到 GitHub

Streamlit Cloud 需要從 GitHub 讀取專案。請建立一個新的 GitHub repository，然後只上傳這些專案檔案：

```text
app.py
requirements.txt
packages.txt
README.md
.streamlit/config.toml
.gitignore
outputs/.gitkeep
work/.gitkeep
```

不要上傳：

```text
.venv/
.idea/
.DS_Store
__pycache__/
```

### 2. 建立 Streamlit Cloud App

到 Streamlit Cloud 建立新 App：

```text
Repository：選你的 GitHub repository
Branch：main
Main file path：app.py
```

展開 Advanced settings，Python version 建議選：

```text
3.11
```

按下 Deploy 後，Streamlit Cloud 會自動安裝 Python 套件與 FFmpeg。

### 3. 常見部署問題

如果部署時卡在 `torch` 安裝，建議刪除 App 後重新 Deploy，並在 Advanced settings 選 Python 3.11。Streamlit Community Cloud 的 Python 版本是透過部署畫面設定，不是透過 `runtime.txt` 設定。

如果出現找不到 FFmpeg，請確認 repository 裡有：

```text
packages.txt
```

內容必須是：

```text
ffmpeg
```

Streamlit Cloud 免費資源有限，處理很大的影片可能會超時或記憶體不足。教學用短影片通常比較適合，若之後要做成正式服務，會更建議改用 Render、Railway、Fly.io 或 VPS。

## 剪輯邏輯

流程如下：

```text
上傳影片
  -> FFprobe 讀取原始長度
  -> FFmpeg 抽取單聲道 16kHz WAV
  -> Silero VAD 偵測 speech timestamps
  -> 合併 gap 小於或等於最短停頓秒數的相鄰 speech 區間
  -> 對保留區間套用前後 padding
  -> FFmpeg trim / atrim / concat 輸出 MP4
```

也就是說，如果兩段語音中間只停了 `0.3` 秒，而最短停頓秒數設定為 `0.8` 秒，這段停頓會保留；如果中間停了 `2.5` 秒，才會剪除中間超過門檻的部分。

## 常見問題

### 找不到 FFmpeg

請確認 `ffmpeg -version` 與 `ffprobe -version` 可以在終端機執行。若 Streamlit 是從 IDE 啟動，也要確認 IDE 繼承得到正確的 PATH。

### 沒有偵測到語音

可以嘗試：

- 降低「VAD 語音判定門檻」
- 確認影片音軌中真的有人聲
- 確認影片不是靜音、沒有音軌，或音量過低

### 輸出影片和原檔格式不同

不論原始檔是 MP4、MOV 或 MKV，輸出都會重新編碼成 H.264 + AAC 的 MP4，方便瀏覽器播放與下載。

## 參考

- Silero VAD 官方 GitHub：https://github.com/snakers4/silero-vad
- Silero VAD PyPI：https://pypi.org/project/silero-vad/
- FFmpeg 官方網站：https://ffmpeg.org/
