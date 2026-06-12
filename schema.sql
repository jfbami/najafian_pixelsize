CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    subfolder_path TEXT NOT NULL,
    account_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    nm_per_pixel REAL,
    calibration_frame TEXT,
    calibration_source TEXT,
    pixels_per_space REAL,
    fft_confidence REAL,
    detector_confidence REAL,
    magnification INTEGER,
    calibration_date TEXT,
    tissue_date TEXT,
    date_delta_days INTEGER,
    agent_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    agent_explanation TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'medium',
    resolved INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS calibration_cache (
    magnification INTEGER NOT NULL,
    calibration_date TEXT NOT NULL,
    account_name TEXT NOT NULL,
    folder_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    nm_per_pixel REAL NOT NULL,
    fft_confidence REAL NOT NULL,
    PRIMARY KEY (magnification, calibration_date, folder_id, frame_index)
);

CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cal_cache_lookup
    ON calibration_cache(magnification, calibration_date);
