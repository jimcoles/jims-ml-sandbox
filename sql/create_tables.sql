--
-- Create our model test results db
--
CREATE TABLE test_result
(
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    datestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
    test_name     TEXT NOT NULL,
    dataset_name  TEXT NOT NULL,
    engine_name   TEXT NOT NULL,
    layer_params  TEXT NOT NULL, -- JSON to describe layers & hyperparams
    time_to_train REAL NOT NULL,
    pearson       REAL NOT NULL
);

CREATE TABLE test_measurement
(
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER NOT NULL,
    name      TEXT    NOT NULL,
    value     REAL    NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES test_result (id),
    CONSTRAINT unique_parent_name UNIQUE (parent_id, name)
);

CREATE TABLE model_layer
(
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    test_result_id INTEGER NOT NULL,
    type           TEXT    NOT NULL,
    FOREIGN KEY (test_result_id) REFERENCES test_result (id)
);

CREATE TABLE layer_param
(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    layer_id        INTEGER NOT NULL,
    parent_param_id INTEGER,
    sequence_index  INTEGER NOT NULL,
    name            TEXT    NOT NULL,
    value           TEXT    NOT NULL,
    FOREIGN KEY (layer_id) REFERENCES model_layer (id),
    FOREIGN KEY (parent_param_id) REFERENCES layer_param (id)
);