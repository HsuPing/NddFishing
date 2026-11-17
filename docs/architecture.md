# 架構與 Phase 1 結論

## 1. 環境檢查結果（2026-11-17 實測）

| 項目 | 結果 |
|---|---|
| 機器 | Apple M3（arm64） |
| macOS | 27.2 |
| Python | 3.13.3（venv）；系統另有 3.11.15、3.14.7 |
| 螢幕 | 內建 2560×1664（scale 2.0）、外接 1920×1080（scale 1.0） |
| 音訊輸入 | 兩支麥克風；沒有 BlackHole 之類的虛擬 loopback 裝置 |
| 螢幕錄製權限 | 已授予（以 Claude 桌面 App 啟動時） |

## 2. 套件相容性

| 用途 | 規格建議 | 採用 | 實測結論 |
|---|---|---|---|
| 畫面擷取 | ScreenCaptureKit / PyObjC | pyobjc-framework-ScreenCaptureKit 13.0 + libdispatch | Phase 2 實測串流、ROI 裁切、執行中改 ROI、停止都正常。 |
| 系統音訊 | SoundCard / SoundDevice | **ScreenCaptureKit**（`capturesAudio`） | 見下方說明。 |
| 裝置音訊 | SoundDevice | sounddevice 0.5.6 | 可列出輸入裝置。作為備援（麥克風或自行安裝的 BlackHole）。 |
| 影像 | OpenCV | opencv-python-headless **5.0.0** | `inRange`、`morphologyEx`、`findContours`（回傳 2 個值）、`connectedComponentsWithStats`、`KalmanFilter`、`matchTemplate` 實測可用。 |
| 數值 | NumPy / SciPy | 2.5.5 / 1.18.1 | 可用，含 `scipy.signal.ShortTimeFFT`。 |
| UI | PySide6 | PySide6-Essentials 6.12.0 | 可匯入。只裝 Essentials（不需要 WebEngine 等 Addons）。 |
| 儲存 | SQLite | 標準函式庫 `sqlite3` | 不需額外套件。 |

### 不採用 SoundCard 的原因

SoundCard 在 macOS 匯入時直接警告 `macOS does not support loopback recording functionality`，`all_microphones(include_loopback=True)` 也只回傳實體麥克風。它無法擷取遊戲輸出的聲音。

sounddevice（PortAudio）同樣只能錄輸入裝置，要錄系統聲音必須另裝 BlackHole 並改系統輸出路由。

ScreenCaptureKit 從 macOS 13 起可以直接擷取系統或單一 App 的音訊，不需要虛擬驅動，而且與畫面共用同一個權限和同一個時鐘，音訊與影像對時最簡單。因此系統音訊以它為主，sounddevice 為備援。

### 選用 opencv-python-headless 的原因

UI 由 PySide6 負責，不需要 OpenCV 自帶的視窗後端；headless 版體積較小，也避免兩套 GUI 後端並存。

## 3. 資料流與執行緒（規劃）

```
ScreenCaptureKit ──frame──▶ 最新一幀槽位 ──▶ 視覺 Worker（Detector → Tracker）──┐
      │                                                                        ├─▶ 狀態機 ─▶ Qt signal ─▶ UI
      └────audio──▶ 環形緩衝 ──▶ 音訊 Worker（特徵 → Classifier）───────────────┘
```

- 擷取 callback 在擷取端自己的執行緒上執行，只做「轉成 NumPy、放進槽位」。
- 視覺 Worker 只取最新一幀；處理不及時丟棄舊幀，不排隊，延遲不會累積。
- 音訊不可丟樣本，使用固定大小的環形緩衝。
- UI 只透過 Qt signal 接收結果，自己不做任何分析。
- 任何 Worker 的例外都轉成錯誤事件送給狀態機（→ `ERROR`），不讓執行緒無聲死亡，也不拖垮程式。

## 4. 介面

| 介面 | 位置 | 說明 |
|---|---|---|
| `FrameSource` / `AudioSource` | `capture/base.py` | `start(callback, on_error, on_end)` / `stop()`。即時擷取與離線重播都實作它。執行緒與停止規則寫在該檔開頭。 |
| `BobberDetector` | `vision/base.py` | `detect(frame) -> Detection \| None`、`reset()`。 |
| `EventClassifier` | `audio/base.py` | `process(chunk) -> list[AudioEvent]`、`reset()`。串流式，自行保留所需歷史。 |

`EventClassifier` 吃原始 `AudioChunk` 而不是固定的特徵結構：未來的 ML 模型可以自選特徵，規則式分類器則在內部使用共用的 `feature_extractor`。

## 5. 重要約定

- **時間戳**：影像與音訊都用主機單調時鐘（秒）。Phase 2 已實測 `CMSampleBuffer` 的 presentation timestamp 與 `time.monotonic()` 同源（差距為數毫秒的擷取延遲）。
- **ROI**：以 0..1 正規化座標儲存，使用時再換算成像素。解析度或 Retina 縮放改變時 ROI 仍然有效。
- **Frame.image**：BGR、已裁成 ROI，建立時即設為唯讀。偵測、錄製、預覽共用同一個陣列，不各自複製。`Frame.origin` 與 `target_size` 記錄它在整個擷取目標中的位置，辨識結果可以換算回螢幕座標。
- **AudioChunk.first_sample_index**：累計樣本數，數字跳號代表掉樣本。
- **設定**：全部在 `config/settings.py`；辨識與狀態機的參數在各自的 Phase 加入。

## 6. 待決事項

- `audio.sample_rate` 目前允許 44100；ScreenCaptureKit 是否接受待 Phase 4 實測，不接受就從設定移除。
- sounddevice 的時間戳不是主機單調時鐘，`device` 音源需在來源端換算。
- （已決定）UI 調整的 ROI 另存 `state.json`，不改寫使用者手寫的 TOML。

## 7. Phase 2：畫面擷取實作筆記

### 實測結論

| 項目 | 結果 |
|---|---|
| 螢幕 ROI 裁切 | 與全畫面對應區域像素完全相同（2x Retina 與 1x 外接螢幕，後者原點為負座標） |
| 視窗 ROI 裁切 | 位置誤差約 1 px，有輕微重取樣 |
| `sourceRect` 座標 | 相對於擷取目標本身（不是全域桌面座標），單位為 point |
| 列緩衝 | 有 padding（400 px 寬 → 每列 1664 bytes），轉換時必須用 `bytesPerRow` |
| 啟動／改 ROI／停止 | 約 80–160 ms／45 ms／3 ms |
| 20 秒實機執行（全螢幕 1920×1248） | 穩定 29 FPS、記憶體不成長、UI 事件迴圈最大間隔 43 ms、CPU 約 50%（單核） |

### PyObjC 的兩個陷阱

1. 從 Objective-C 呼叫的 Python callback 若拋出例外，或 completion handler 回傳非 `None`，整個程式會被終止。所以 `screen_capture.py` 的每個 callback 都有完整防護且只回傳 `None`。
2. 沒有 `NSApplication` 的程式（測試、命令列工具）直接建立視窗 filter 會因 `CGS_REQUIRE_INIT` 中止。先呼叫一次 `CGMainDisplayID()` 即可，有回歸測試。

### 執行緒

```
ScreenCaptureKit dispatch queue          UI 執行緒
  sample buffer                            QTimer 15 ms
   → 鎖定像素緩衝、BGRA→BGR（唯一一次複製）    → LatestSlot.take()
   → Frame                                  → VideoWidget 以 QImage 包住同一塊記憶體繪製
   → LatestSlot.put()（覆蓋未取走的舊幀）
```

- `ScreenCaptureSource` 用一把鎖保護狀態，並在持有鎖時呼叫使用者 callback，因此 `stop()` 返回後保證不再有 callback。代價是 callback 內不可再呼叫該來源的任何方法；來源會偵測並拋出 `CaptureError`，不會卡死。
- delegate 只以弱參考持有來源，另有 finalizer：來源物件沒 `stop()` 就被丟棄時，系統串流仍會被停止。
- `stop()` 與正在等待系統回應的 `start()` 競爭時，`start()` 會拋出 `CaptureError`，不會留下「看似啟動但沒有畫面」的來源。
- 改 ROI 期間暫停送出畫面，避免畫面內容與 ROI 標記不一致；逾時則視為擷取失敗。
- 擷取失敗（例如螢幕被拔除）經 Qt 的 queued signal 回到 UI 執行緒處理。
- 畫面靜止時 ScreenCaptureKit 只送 idle 通知，這些會被略過，所以實測 FPS 會低於設定值。

### ROI 流程

未框選時擷取整個目標並由系統縮到 `max_width` 以內；框選後以 `updateConfiguration` 直接改擷取範圍，不重啟串流，ROI 內為原生解析度。每個 `Frame` 帶著自己涵蓋的 `roi`，所以在已裁切的畫面上再框選也能換算回整個目標的座標。

### Phase 2 Code Review 未處理項目

| 項目 | 原因 | 預定 |
|---|---|---|
| 列出來源／啟動／停止／改 ROI 在 UI 執行緒上執行 | 典型耗時 0.05–0.2 秒且只在使用者點擊時發生；改成背景執行需要重寫控制流程 | Phase 6 |
| 視窗被縮放後不會重新量測尺寸 | 需要實際縮放視窗才能驗證每幀附帶的尺寸資訊，目前無法在不操作視窗的情況下測試 | Phase 3（開始用實際遊戲視窗時） |
