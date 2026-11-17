# WoW Forever Fishing Research（macOS）

以視覺辨識與音訊分析研究魔獸世界釣魚流程的模組化工具。

**研究範圍**：畫面擷取、魚標辨識與追蹤、釣魚音效辨識、模擬狀態機、即時監控、準確率分析。

**不做的事**：不送出任何鍵盤／滑鼠輸入到遊戲、不讀寫記憶體、不處理封包。狀態機只更新研究狀態。

## 目前進度

| Phase | 內容 | 狀態 |
|---|---|---|
| 1 | 環境檢查、專案架構、模組介面 | 完成 |
| 2 | 畫面擷取、預覽、ROI、FPS | 完成 |
| 3 | 魚標 Detector、追蹤器、離線測試 | 進行中（錄製與離線重播已完成，辨識待實際素材） |
| 4 | 音訊擷取、RMS/FFT、事件分類 | 未開始 |
| 5 | 釣魚狀態機 | 未開始 |
| 6 | UI 整合、效能、長時間測試 | 未開始 |

目前**可執行**：環境檢查、畫面擷取與即時預覽、ROI 框選、FPS／延遲／CPU 統計、畫面樣本錄製與離線重播。魚標辨識、音訊、狀態機尚未實作。

## 安裝

需求：macOS 13 以上、Python 3.11 以上。

```bash
uv venv --python 3.13 .venv
```

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

沒有 `uv` 時：`python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`。

## 啟動

先確認環境：

```bash
.venv/bin/python -m wowfishing --check
```

每一行 `[OK]` 代表該項在本機實測通過，`[!!]` 代表需要處理。

開啟監控視窗：

```bash
.venv/bin/python -m wowfishing
```

## 操作

1. 在「擷取來源」選螢幕或視窗，設定 FPS，按「開始研究」。
2. 按「框選 ROI」，在預覽上拖曳出要觀察的水面範圍。擷取會立刻改成只抓這個範圍。
3. 要重新框選時按「顯示全畫面」回到完整畫面。
4. ROI 會存到 `state.json`，下次啟動自動沿用。

### 錄製魚標樣本

遊戲請用視窗模式或「全螢幕（視窗化）」，放在一個螢幕；監控視窗放另一個螢幕。擷取來源選「螢幕」。

1. 開始擷取後，框選 ROI：只框魚標會落下的那片水面，越小越好。
2. FPS 設 15 就夠。
3. 拋竿前按「開始錄製」，收竿後按「停止錄製」。一段錄一次拋竿到上鉤的完整過程。
4. 檔案存在 `recordings/日期-時間/`：`frames/*.png`（無損）、`frames.jsonl`（每張的時間戳與 ROI）、`meta.json`。

實測檔案大小：1024×666 的 ROI 每張約 0.2 MB（一般桌面畫面；遊戲畫面細節多，會更大）。全螢幕 30 FPS 約每秒 25 MB，而且磁碟會逐漸跟不上，所以務必先框 ROI。

磁碟來不及寫時會略過畫面並在介面上顯示略過張數，不會拖慢擷取。

統計欄位：

| 欄位 | 意義 |
|---|---|
| FPS | 實際收到的新畫面張數。畫面靜止時系統不送新畫面，數字會低於設定值，這是正常的。 |
| 擷取延遲 | 畫面顯示到程式收到之間的時間。 |
| 畫面大小 | 目前每張畫面的像素尺寸。 |
| 未顯示幀 | 預覽來不及畫而略過的張數（不影響擷取）。 |
| CPU | 本程式佔用，100% 等於一顆核心。 |

## 權限

畫面與系統音訊擷取都走 ScreenCaptureKit，只需要一項權限：

系統設定 → 隱私權與安全性 → **螢幕與系統音訊錄製** → 勾選「啟動 Python 的那個 App」（Terminal、iTerm、VS Code 等）。勾選後要重開該 App。

權限屬於啟動 Python 的 App，不是 Python 本身。換一個終端機執行就要重新授權。

## 設定

所有參數集中在 [src/wowfishing/config/settings.py](src/wowfishing/config/settings.py)，可用 TOML 覆寫：

```toml
[capture]
fps = 20
window_owner = "World of Warcraft"
roi = { x = 0.25, y = 0.2, width = 0.5, height = 0.5 }   # 0..1 正規化座標

[audio]
source = "system"        # 或 "device"（搭配 device_name）
```

`capture.max_width`（預設 1920）：畫面寬度超過這個值時由系統縮小。框選 ROI 後通常不會超過，所以 ROI 內是原生解析度。

```bash
.venv/bin/python -m wowfishing --config config.local.toml --check
```

未知的區段或鍵會直接報錯，避免打錯字卻默默使用預設值。

## 架構

```
src/wowfishing/
├── main.py              CLI 進入點
├── models.py            跨模組共用資料型別（Roi、Frame、Detection、AudioChunk、AudioEvent）
├── config/settings.py   集中參數與 TOML 載入
├── config/state.py      UI 改動的值（ROI）存成 state.json
├── capture/base.py      FrameSource / AudioSource 介面（即時擷取與離線重播共用）
├── capture/screen_capture.py   ScreenCaptureKit 即時擷取
├── capture/geometry.py  ROI→擷取範圍換算、像素緩衝轉換
├── capture/recorder.py  畫面錄成無損 PNG 序列
├── capture/replay.py    讀回錄製資料（批次或依原始時間重播）
├── capture/targets.py   可擷取的螢幕／視窗描述與選擇
├── vision/base.py       BobberDetector 介面
├── audio/base.py        EventClassifier 介面
├── state/               模擬狀態機（Phase 5）
├── dashboard/           PySide6 監控介面（主視窗、預覽元件、擷取控制）
├── storage/             SQLite：樣本、結果、人工標記（尚未實作）
└── utils/               logger、環境檢查、FPS／延遲統計、最新一幀槽位
```

依賴方向：`models` ← `capture` / `vision` / `audio` ← `state` ← `dashboard`。`vision` 與 `audio` 互不依賴，也不依賴擷取實作，因此可以直接吃錄製檔做離線測試。

設計細節與套件相容性結論見 [docs/architecture.md](docs/architecture.md)。

## 開發

```bash
.venv/bin/pytest
```

標記為 `capture` 的測試會真的啟動 ScreenCaptureKit；沒有螢幕錄製權限時自動略過。

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy
```

## 已知限制

- 尚未用實際遊戲畫面或音效驗證任何辨識功能（辨識模組尚未實作）。
- 僅在 Apple Silicon（M3、macOS 27.2、Python 3.13）實測；Intel Mac 與 macOS 13 的縮放比例備援路徑未驗證。
- 螢幕擷取的 ROI 裁切經像素比對完全一致（Retina 2x 與外接 1x 螢幕）。**視窗**擷取的裁切位置誤差約 1 像素，且有輕微重取樣。
- 擷取視窗時若視窗被縮放，畫面尺寸不會跟著更新，需停止後重新開始。
- 按「開始研究」會讓介面停頓約 0.1–0.2 秒、套用 ROI 約 0.05 秒（等待系統回應）；系統無回應時最久會等 5 秒逾時。擷取進行中不會阻塞介面。移到背景執行緒排在 Phase 6。
- 全螢幕 1920×1248、30 FPS 預覽約佔一顆核心的 50%；框選 ROI 後畫面變小，負擔隨之下降。
- 長時間執行只測過 20 秒（記憶體穩定）；數小時等級的測試在 Phase 6。
