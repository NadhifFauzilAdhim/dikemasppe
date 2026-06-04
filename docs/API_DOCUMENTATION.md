# PPE Violation Upload API - Documentation

Dokumentasi API endpoint untuk menerima data pelanggaran PPE dari sistem deteksi CCTV.

---

## Overview

Sistem deteksi PPE akan mengirim data pelanggaran ke backend setiap kali terdeteksi pekerja **tanpa APD** (NO-Hardhat, NO-Mask, NO-Safety Vest). Data yang dikirim berupa **foto capture pelanggaran** beserta **metadata JSON** yang berisi informasi detail.

---

## Endpoint

```
POST /api/v1/violations
```

### Headers

| Header          | Tipe   | Required | Keterangan                     |
|-----------------|--------|----------|--------------------------------|
| `Authorization` | string | Optional | `Bearer <api_key>` jika diset  |
| `Content-Type`  | -      | Auto     | `multipart/form-data` (auto)   |

---

## Request Format

**Content-Type:** `multipart/form-data`

### Form Fields

| Field     | Tipe   | Required | Keterangan                           |
|-----------|--------|----------|--------------------------------------|
| `image`   | file   | Yes      | File JPEG capture frame pelanggaran  |
| `payload` | string | Yes      | JSON string berisi metadata violasi  |

### Payload JSON Structure

```json
{
    "timestamp": "2026-06-03T22:15:30.123456",
    "camera_id": "CAM-001",
    "violation_type": "NO-Hardhat",
    "violation_class_id": 2,
    "confidence": 0.8723,
    "bbox": {
        "x1": 234,
        "y1": 156,
        "x2": 389,
        "y2": 445
    },
    "person_count": 3,
    "all_detections": [
        {
            "class_id": 2,
            "class_name": "NO-Hardhat",
            "confidence": 0.8723,
            "bbox": {"x1": 234, "y1": 156, "x2": 389, "y2": 445},
            "area": 44805,
            "center": {"x": 311, "y": 300}
        },
        {
            "class_id": 5,
            "class_name": "Person",
            "confidence": 0.9156,
            "bbox": {"x1": 200, "y1": 100, "x2": 420, "y2": 520},
            "area": 92400,
            "center": {"x": 310, "y": 310}
        },
        {
            "class_id": 7,
            "class_name": "Safety Vest",
            "confidence": 0.7834,
            "bbox": {"x1": 215, "y1": 200, "x2": 400, "y2": 450},
            "area": 46250,
            "center": {"x": 307, "y": 325}
        }
    ],
    "frame_id": 1520,
    "inference_time_ms": 23.45
}
```

### Payload Field Reference

| Field                | Tipe     | Keterangan                                          |
|----------------------|----------|-----------------------------------------------------|
| `timestamp`          | string   | ISO 8601 timestamp saat pelanggaran terdeteksi       |
| `camera_id`          | string   | ID kamera yang mendeteksi (configurable)             |
| `violation_type`     | string   | Jenis pelanggaran: `NO-Hardhat`, `NO-Mask`, atau `NO-Safety Vest` |
| `violation_class_id` | integer  | Class ID model: `2` = NO-Hardhat, `3` = NO-Mask, `4` = NO-Safety Vest |
| `confidence`         | float    | Confidence score deteksi (0.0 - 1.0)                |
| `bbox`               | object   | Bounding box pelanggaran dalam piksel                |
| `bbox.x1`            | integer  | Koordinat kiri atas X                                |
| `bbox.y1`            | integer  | Koordinat kiri atas Y                                |
| `bbox.x2`            | integer  | Koordinat kanan bawah X                              |
| `bbox.y2`            | integer  | Koordinat kanan bawah Y                              |
| `person_count`       | integer  | Jumlah orang terdeteksi di frame                     |
| `all_detections`     | array    | Semua deteksi dalam frame (termasuk APD yang dipakai)|
| `frame_id`           | integer  | Nomor frame saat pelanggaran terdeteksi              |
| `inference_time_ms`  | float    | Waktu inferensi model dalam milliseconds             |

### Image File

| Property    | Detail                                           |
|-------------|--------------------------------------------------|
| Format      | JPEG                                             |
| Quality     | 85%                                              |
| Resolution  | Sesuai resolusi kamera (default 1280x720)        |
| Content     | Frame yang sudah di-annotasi dengan bounding box |
| Filename    | `violation_{type}_{timestamp}.jpg`               |

---

## Response Format

### Success (200 / 201)

```json
{
    "status": "success",
    "message": "Violation recorded successfully",
    "data": {
        "id": 1234,
        "violation_type": "NO-Hardhat",
        "camera_id": "CAM-001",
        "created_at": "2026-06-03T22:15:31.000Z"
    }
}
```

### Error (4xx / 5xx)

```json
{
    "status": "error",
    "message": "Detailed error description",
    "errors": {}
}
```

### Status Codes

| Code | Keterangan                           |
|------|--------------------------------------|
| 200  | Violation berhasil disimpan          |
| 201  | Violation berhasil dibuat            |
| 400  | Bad request / payload tidak valid    |
| 401  | Unauthorized (API key salah/expired) |
| 413  | Image terlalu besar                  |
| 422  | Validation error                     |
| 500  | Internal server error                |
| 503  | Service unavailable                  |

---

## Class ID Reference

| ID | Class Name       | Kategori    | Keterangan                  |
|----|------------------|-------------|-----------------------------|
| 0  | Hardhat          | APD OK      | Memakai helm                |
| 1  | Mask             | APD OK      | Memakai masker              |
| 2  | NO-Hardhat       | VIOLATION   | Tidak memakai helm          |
| 3  | NO-Mask          | VIOLATION   | Tidak memakai masker        |
| 4  | NO-Safety Vest   | VIOLATION   | Tidak memakai rompi safety  |
| 5  | Person           | Object      | Orang terdeteksi            |
| 6  | Safety Cone      | Object      | Kerucut safety              |
| 7  | Safety Vest      | APD OK      | Memakai rompi safety        |
| 8  | machinery        | Object      | Mesin/alat berat            |
| 9  | vehicle          | Object      | Kendaraan                   |

> **Note:** Hanya class dengan kategori **VIOLATION** (ID 2, 3, 4) yang akan di-upload ke server.

---

## cURL Example

```bash
curl -X POST http://localhost:8000/api/v1/violations \
  -H "Authorization: Bearer your-api-key-here" \
  -F "image=@violation_NO-Hardhat_2026-06-03T22-15-30.jpg" \
  -F 'payload={
    "timestamp": "2026-06-03T22:15:30.123456",
    "camera_id": "CAM-001",
    "violation_type": "NO-Hardhat",
    "violation_class_id": 2,
    "confidence": 0.8723,
    "bbox": {"x1": 234, "y1": 156, "x2": 389, "y2": 445},
    "person_count": 3,
    "all_detections": [],
    "frame_id": 1520,
    "inference_time_ms": 23.45
  }'
```

---

## Backend Implementation Notes

### Database Schema (Suggestion)

```sql
CREATE TABLE violations (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp       DATETIME NOT NULL,
    camera_id       VARCHAR(50) NOT NULL,
    violation_type  VARCHAR(50) NOT NULL,
    confidence      DECIMAL(5,4) NOT NULL,
    bbox_x1         INT,
    bbox_y1         INT,
    bbox_x2         INT,
    bbox_y2         INT,
    person_count    INT DEFAULT 0,
    image_path      VARCHAR(500),
    raw_detections  JSON,
    frame_id        INT,
    inference_ms    DECIMAL(8,2),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_camera (camera_id),
    INDEX idx_type (violation_type),
    INDEX idx_timestamp (timestamp)
);
```

### Laravel Controller Skeleton

```php
// routes/api.php
Route::post('/v1/violations', [ViolationController::class, 'store']);

// ViolationController.php
public function store(Request $request)
{
    $request->validate([
        'image'   => 'required|image|max:5120',  // Max 5MB
        'payload' => 'required|json',
    ]);

    $payload = json_decode($request->input('payload'), true);

    // Store image
    $path = $request->file('image')->store('violations', 'public');

    // Save to database
    $violation = Violation::create([
        'timestamp'      => $payload['timestamp'],
        'camera_id'      => $payload['camera_id'],
        'violation_type' => $payload['violation_type'],
        'confidence'     => $payload['confidence'],
        'bbox_x1'        => $payload['bbox']['x1'],
        'bbox_y1'        => $payload['bbox']['y1'],
        'bbox_x2'        => $payload['bbox']['x2'],
        'bbox_y2'        => $payload['bbox']['y2'],
        'person_count'   => $payload['person_count'],
        'image_path'     => $path,
        'raw_detections' => json_encode($payload['all_detections']),
        'frame_id'       => $payload['frame_id'],
        'inference_ms'   => $payload['inference_time_ms'],
    ]);

    return response()->json([
        'status'  => 'success',
        'message' => 'Violation recorded successfully',
        'data'    => $violation,
    ], 201);
}
```

---

## Configuration (config.yaml)

```yaml
api:
  enabled: true                     # Aktifkan upload
  base_url: "http://your-server.com"
  endpoint: "/api/v1/violations"
  api_key: "your-secret-key"        # Optional Bearer token
  camera_id: "CAM-001"              # ID kamera ini
  timeout: 10                       # Timeout request (detik)
  max_retries: 3                    # Retry jika gagal
  cooldown_seconds: 30              # Jeda antar upload violation yang sama
  save_local: true                  # Simpan capture lokal juga
  capture_dir: "outputs/violations" # Folder simpan lokal
```

### Cooldown Logic

Untuk menghindari spam upload dari frame berturut-turut:
- Setiap **jenis** violation (NO-Hardhat, NO-Mask, NO-Safety Vest) punya cooldown terpisah
- Default **30 detik** antara upload untuk jenis violation yang sama
- Jika ada NO-Hardhat di frame 100 dan frame 101, hanya frame 100 yang di-upload
- Jika ada NO-Hardhat DAN NO-Mask di frame yang sama, keduanya di-upload (beda tipe)

---

## Flow Diagram

```
[Camera/RTSP] → [YOLOv8 Detection] → [Postprocessor]
                                           │
                                    Has NO-* violation?
                                     │            │
                                    YES           NO
                                     │            │
                              Cooldown OK?     (skip)
                                     │
                                    YES
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                   [Save Local JPEG]    [Queue for Upload]
                                              │
                                    [Background Thread]
                                              │
                                  POST /api/v1/violations
                                     (image + JSON)
                                              │
                                    [Backend Server]
                                              │
                                    [Dashboard Display]
```
