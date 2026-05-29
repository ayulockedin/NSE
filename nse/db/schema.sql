-- NSE logging & audit schema (build-ready blueprint section 5.4).
-- SQLite dialect: SERIAL -> INTEGER PRIMARY KEY AUTOINCREMENT, JSON -> TEXT.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS branches (
    id           TEXT PRIMARY KEY,
    task_id      TEXT NOT NULL,
    planner_json TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id  TEXT REFERENCES branches(id),
    p_t        REAL,
    u          REAL,
    r_critic   REAL,
    r_long     REAL,
    c_planner  REAL,
    p_c        INTEGER,            -- 0 or 1
    score      REAL,
    routing    TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS outcomes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id    TEXT REFERENCES branches(id),
    compiled     INTEGER,          -- 0/1
    tests_passed INTEGER,          -- 0/1
    logs         TEXT,
    runtime      REAL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pruned_branches (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id         TEXT,
    reason            TEXT,
    sampled_for_audit INTEGER DEFAULT 0,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS calibrations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ece        REAL,
    brier      REAL,
    action     TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Outcomes of re-running sampled pruned branches (false-negative hunting).
CREATE TABLE IF NOT EXISTS audit_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    pruned_id      INTEGER REFERENCES pruned_branches(id),
    branch_id      TEXT,
    tests_passed   INTEGER,    -- 0/1 when re-run; NULL if not re-executable
    false_negative INTEGER,    -- 1 if pruned for low score but actually passed
    runtime        REAL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_branch ON predictions(branch_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_branch    ON outcomes(branch_id);
CREATE INDEX IF NOT EXISTS idx_pruned_sampled     ON pruned_branches(sampled_for_audit);
