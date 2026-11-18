-- unit-matcher PostgreSQL schema. Every statement is idempotent.
--
--   1) reference data   loaded from data/*.json by scripts/seed_postgres.py
--   2) run output       resolved records, review queue, hierarchy paths
--   3) review / RAG     decisions, learned tables, vector store

-- ============================================================
-- 1) Referans / kanonik veri
-- ============================================================

CREATE TABLE IF NOT EXISTS canonical_units (
    uuid        UUID PRIMARY KEY,
    ad          TEXT NOT NULL,
    tur         TEXT,
    kod         TEXT,
    ust_uuid    UUID REFERENCES canonical_units(uuid) DEFERRABLE INITIALLY DEFERRED,
    ust_kod     TEXT,
    node_id     TEXT[] NOT NULL DEFAULT '{}',
    kaynak      TEXT
);
CREATE INDEX IF NOT EXISTS idx_canonical_units_ad       ON canonical_units (upper(ad));
CREATE INDEX IF NOT EXISTS idx_canonical_units_kod      ON canonical_units (kod);
CREATE INDEX IF NOT EXISTS idx_canonical_units_ust_uuid ON canonical_units (ust_uuid);
CREATE INDEX IF NOT EXISTS idx_canonical_units_node_id  ON canonical_units USING GIN (node_id);

-- synonym -> official name
CREATE TABLE IF NOT EXISTS synonyms (
    alt_isim   TEXT PRIMARY KEY,
    resmi_isim TEXT NOT NULL,
    kanonik_id TEXT
);

-- abbreviation -> expansion
CREATE TABLE IF NOT EXISTS abbreviations (
    kisaltma TEXT PRIMARY KEY,
    resmi_ad TEXT NOT NULL
);

-- per-unit ID by staff category, '- (ATANAMAZ)' when not allowed
CREATE TABLE IF NOT EXISTS kadro_kurallari (
    birim_adi         TEXT PRIMARY KEY,
    idari_id          TEXT,
    diger_personel_id TEXT,
    akademik_id       TEXT
);

-- ============================================================
-- 2) Run output (same definitions as engine/db_sink.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS eslesen_kayitlar (
    id                  BIGSERIAL PRIMARY KEY,
    kaynak              TEXT NOT NULL,
    record_id           TEXT NOT NULL,
    dominant_branch     TEXT,
    overall_confidence  NUMERIC(4,2),
    fields              JSONB NOT NULL,
    guncellenme_zamani  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kaynak, record_id)
);
CREATE INDEX IF NOT EXISTS idx_eslesen_kayitlar_fields ON eslesen_kayitlar USING GIN (fields);

-- records waiting for a person, stored unchanged
CREATE TABLE IF NOT EXISTS human_review_kayitlari (
    id                  BIGSERIAL PRIMARY KEY,
    kaynak              TEXT NOT NULL,
    record_id           TEXT NOT NULL,
    dominant_branch     TEXT,
    overall_confidence  NUMERIC(4,2),
    fields              JSONB NOT NULL,
    guncellenme_zamani  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kaynak, record_id)
);
CREATE INDEX IF NOT EXISTS idx_human_review_kayitlari_fields ON human_review_kayitlari USING GIN (fields);

-- full path from the root for every unit, derived from canonical_units
CREATE TABLE IF NOT EXISTS birim_hiyerarsi (
    uuid               UUID PRIMARY KEY REFERENCES canonical_units(uuid),
    ad                 TEXT NOT NULL,
    kod                TEXT,
    zincir             JSONB NOT NULL,
    zincir_metni       TEXT NOT NULL,
    guncellenme_zamani TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 3) Human review and RAG (engine/feedback.py, engine/vector_store.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS human_decisions_log (
    id                  BIGSERIAL PRIMARY KEY,
    "timestamp"         TIMESTAMPTZ NOT NULL,
    reviewer            TEXT NOT NULL,
    record_id           TEXT NOT NULL,
    field_name          TEXT NOT NULL,
    karar_tipi          TEXT NOT NULL CHECK (karar_tipi IN ('metin_eslesme', 'id_eslesme')),
    incoming_text       TEXT,
    incoming_id         TEXT,
    normalized_text     TEXT,
    onceki_confidence   NUMERIC(4,2),
    onceki_reason       TEXT,
    secilen_uuid        UUID,
    secilen_ad          TEXT,
    secilen_kod         TEXT,
    not_                TEXT
);
-- kaynak was added later; older rows have NULL there
ALTER TABLE human_decisions_log ADD COLUMN IF NOT EXISTS kaynak TEXT;
CREATE INDEX IF NOT EXISTS idx_human_decisions_log_record ON human_decisions_log (record_id, field_name);
CREATE INDEX IF NOT EXISTS idx_human_decisions_log_norm   ON human_decisions_log (normalized_text);

-- text channel of promote_to_learned_corrections()
CREATE TABLE IF NOT EXISTS learned_corrections (
    normalized_text  TEXT PRIMARY KEY,
    resmi_isim       TEXT,
    kanonik_id       TEXT,
    kanonik_uuid     UUID,
    kaynak           TEXT,
    onay_sayisi      INTEGER NOT NULL DEFAULT 1,
    son_karar_tarihi TIMESTAMPTZ
);

-- ID channel of promote_to_learned_corrections()
CREATE TABLE IF NOT EXISTS learned_id_aliases (
    incoming_id      TEXT PRIMARY KEY,
    resmi_isim       TEXT,
    kanonik_id       TEXT,
    kanonik_uuid     UUID,
    kaynak           TEXT,
    onay_sayisi      INTEGER NOT NULL DEFAULT 1,
    son_karar_tarihi TIMESTAMPTZ
);

-- feedback.report_missing_canonical(); kapat* columns track who closed it
CREATE TABLE IF NOT EXISTS kanonik_eksik_bildirimleri (
    id               BIGSERIAL PRIMARY KEY,
    "timestamp"      TIMESTAMPTZ NOT NULL,
    reviewer         TEXT NOT NULL,
    record_id        TEXT NOT NULL,
    field_name       TEXT NOT NULL,
    incoming_text    TEXT,
    incoming_id      TEXT,
    normalized_text  TEXT,
    onceki_reason    TEXT,
    not_             TEXT,
    kapatildi_mi     BOOLEAN NOT NULL DEFAULT false,
    kapatan          TEXT,
    kapatma_tarihi   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS vector_index (
    normalized_text TEXT PRIMARY KEY,
    kanonik_uuid    UUID,
    resmi_isim      TEXT,
    kanonik_kod     TEXT,
    layer           TEXT,
    confidence      NUMERIC(4,2),
    kaynak          TEXT,
    record_id       TEXT
);
