-- Enable fuzzy text matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Elections table (one row per JSON file)
CREATE TABLE elections (
    id SERIAL PRIMARY KEY,
    record_id VARCHAR(255) NOT NULL UNIQUE,
    year SMALLINT NOT NULL,
    office VARCHAR(100) NOT NULL,
    stage VARCHAR(100) NOT NULL,
    total_votes INTEGER NOT NULL
);

-- Districts table (one row per district within an election)
CREATE TABLE districts (
    id SERIAL PRIMARY KEY,
    election_id INTEGER NOT NULL REFERENCES elections(id) ON DELETE CASCADE,
    district_name VARCHAR(100) NOT NULL,
    total_votes INTEGER NOT NULL,
    win_number NUMERIC,
    flip_number NUMERIC,
    win_gap NUMERIC
);

-- Precincts table (one row per precinct within a district)
CREATE TABLE precincts (
    id SERIAL PRIMARY KEY,
    district_id INTEGER NOT NULL REFERENCES districts(id) ON DELETE CASCADE,
    precinct_name VARCHAR(255) NOT NULL,
    total_votes INTEGER NOT NULL,
    win_number NUMERIC,
    flip_number NUMERIC,
    win_gap NUMERIC,
    county VARCHAR(100),
    precinct_code VARCHAR(50),
    precinct_label VARCHAR(255)
);

-- Results table (one row per candidate per district or precinct)
CREATE TABLE results (
    id SERIAL PRIMARY KEY,
    district_id INTEGER REFERENCES districts(id) ON DELETE CASCADE,
    precinct_id INTEGER REFERENCES precincts(id) ON DELETE CASCADE,
    candidate_name VARCHAR(255) NOT NULL,
    votes INTEGER NOT NULL,
    CONSTRAINT results_parent_check CHECK (
        (district_id IS NOT NULL AND precinct_id IS NULL) OR
        (district_id IS NULL AND precinct_id IS NOT NULL)
    )
);

-- Performance indexes
CREATE INDEX idx_elections_record_id ON elections(record_id);
CREATE INDEX idx_elections_office ON elections(office);
CREATE INDEX idx_elections_year ON elections(year);
CREATE INDEX idx_districts_district_name ON districts(district_name);
CREATE INDEX idx_districts_election_id ON districts(election_id);
CREATE INDEX idx_precincts_district_id ON precincts(district_id);
CREATE INDEX idx_results_district_id ON results(district_id);
CREATE INDEX idx_results_precinct_id ON results(precinct_id);

-- Trigram index for fuzzy precinct name search
CREATE INDEX idx_precincts_name_trgm ON precincts USING gin (precinct_name gin_trgm_ops);
