---
applyTo: '**'
---

# CS:GO Analytics Project Instructions

Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

## Project Overview
This project analyzes Counter-Strike: Global Offensive (CS:GO) match data to create round-level and match-level statistics. The data comes from parsed demo files and HLTV match information, stored in PostgreSQL for analysis using Jupyter notebooks.

**Primary Goals:**
- Generate round-level economic and performance statistics
- Create match-level aggregated insights
- Correlate team performance with economic decisions
- Analyze competitive patterns across professional matches

## Architecture & Data Flow
- **Data Source**: PostgreSQL database containing parsed CS:GO demo data and HLTV metadata
- **Analysis Layer**: Jupyter notebooks for exploratory data analysis and statistical modeling
- **Output**: Statistical insights, visualizations, and predictive models 

## Database Connection Standards
Always use this connection pattern for PostgreSQL access:
```python
DB_CONFIG = {
    "dbname": "csgo_parsed",
    "user": "csgo_parser",
    "password": "3?6B7yTGPrkJF34p",
    "host": "192.168.1.100",
    "port": "5444"
}
connection = psycopg2.connect(**DB_CONFIG)
```

## Core Data Tables & Relationships

### Primary Analysis Tables
- **`rounds_ed`**: Core round information (2.2M rows) - match outcomes, sides, overtime
- **`player_round_ed`**: Player performance per round - economic and statistical data
- **`kills_round_ed`**: Individual kill events (14.9M rows) - weapons, positions, timing
- **`inventory_round_ed`**: Equipment tracking (235M rows) - purchases, drops, usage
- **`bomb_events_round_ed`**: Bomb plant/defuse events (13M rows)

### Match Metadata Tables  
- **`hltv_match_info`**: Match details, teams, dates (78K matches)
- **`hltv_match_map`**: Map-specific scores and results (154K maps)
- **`hltv_events`**: Tournament/event information (6.5K events)

### Key Relationships (No formal FKs, but logical connections)
- `rounds_ed.match_id` → `hltv_match_info.match_id`
- `rounds_ed.id` → `kills_round_ed.round_id`
- `rounds_ed.id` → `player_round_ed.round_id`
- `hltv_match_info.event_id` → `hltv_events.event_id`

## Data Schema Conventions

### Team Identification
- **Team 1 vs Team 2**: Use `team1_winner` (boolean) in `rounds_ed`
- **CT vs T Sides**: Use `ct_winner` and `is_ct_t1` for side analysis
- **HLTV Teams**: `team_1_id`/`team_2_id` in `hltv_match_info`

### Round Structure
- **Regular Rounds**: `round_num` 1-30 (15 per side)
- **Overtime**: `is_ot = true`, continues numbering beyond 30
- **Side Switches**: `is_side_switch = true` marks half-time and OT switches

### Economic Data Patterns
Economic analysis requires joining `player_round_ed` with `rounds_ed`:
```sql
-- Standard economic aggregation pattern
SELECT 
    r.id as round_id,
    r.match_id,
    r.round_num,
    r.team1_winner,
    -- Aggregate player economic data
    SUM(CASE WHEN p.team_num = 1 THEN p.money_spent ELSE 0 END) as team1_spent,
    SUM(CASE WHEN p.team_num = 2 THEN p.money_spent ELSE 0 END) as team2_spent
FROM rounds_ed r
JOIN player_round_ed p ON r.id = p.round_id
GROUP BY r.id, r.match_id, r.round_num, r.team1_winner
```

## Common Analysis Patterns

### 1. Round-Level Economic Analysis
```python
# Calculate team spending differences
def calculate_team_economics(round_data):
    round_data['spending_diff'] = round_data['team1_spent'] - round_data['team2_spent']
    round_data['winner_binary'] = round_data['team1_winner'].astype(int)
    return round_data
```

### 2. Match-Level Aggregation
```python
# Aggregate round data to match level
def aggregate_to_match_level(round_data):
    match_stats = round_data.groupby('match_id').agg({
        'team1_winner': 'sum',  # Team 1 round wins
        'spending_diff': 'mean',  # Average spending advantage
        'round_num': 'max'  # Total rounds played
    })
    return match_stats
```

### 3. Side-Based Analysis
```python
# Analyze CT vs T performance
def analyze_side_performance(round_data):
    # Team 1 as CT: is_ct_t1 = True and team1_winner = True
    # Team 1 as T: is_ct_t1 = False and team1_winner = True
    ct_performance = round_data.groupby(['match_id', 'is_ct_t1']).agg({
        'team1_winner': 'mean'  # Win rate by side
    })
    return ct_performance
```

## Statistical Analysis Standards

### Data Cleaning Pipeline
1. **Remove invalid rounds**: Filter out rounds with missing critical data
2. **Handle economic outliers**: Cap extreme spending values (>$50,000)
3. **Normalize team positions**: Ensure consistent team1/team2 assignment
4. **Binary encoding**: Convert boolean outcomes to 0/1 for modeling

### Standard Libraries & Imports
```python
# Core data manipulation
import pandas as pd
import numpy as np
import psycopg2

# Statistical analysis
from scipy.stats import pointbiserialr, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Visualization (when needed)
import matplotlib.pyplot as plt
import seaborn as sns
```

### Model Evaluation Patterns
- Use `train_test_split(test_size=0.2, random_state=42)` for reproducibility
- Report both accuracy and correlation coefficients
- Always check for multicollinearity in economic features
- Validate models across different time periods/events

## CS:GO Domain Knowledge

### Economic System
- **Starting Money**: $800 CT, $400 T at round start
- **Max Money**: $16,000 per player
- **Loss Bonus**: Increases with consecutive losses (up to $3,400)
- **Weapon Costs**: Rifles ~$2,700-3,100, AWP $4,750, Pistols $200-700

### Round Types (for categorization)
- **Eco Rounds**: Low spending (<$1,500 per player average)
- **Force Buy**: Medium spending on losing streak
- **Full Buy**: High spending (>$3,000 per player average)
- **Save Round**: Minimal spending to preserve economy

### Map Knowledge
- **Bomb Sites**: A and B sites for planted bomb scenarios
- **Sides Matter**: Some maps favor CT or T side significantly
- **Overtime**: First to win 4 rounds in OT (switch sides every 3 rounds)

## Notebook Development Guidelines

### File Naming Convention
- `basic_analytics_v*.ipynb`: Core economic correlation analysis
- `distribution_analytics_v*.ipynb`: Spending distribution and categorization
- `match_level_v*.ipynb`: Match-aggregated statistics
- `predictive_v*.ipynb`: Machine learning models

### Cell Structure Standards
1. **Setup**: Imports, database connection, configuration
2. **Data Loading**: SQL queries with proper error handling
3. **Data Cleaning**: Standardized preprocessing pipeline
4. **Analysis**: Statistical tests, correlations, visualizations
5. **Modeling**: Train/test splits, model fitting, evaluation
6. **Conclusions**: Summary statistics and insights

### Error Handling
```python
# Always wrap database operations
try:
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
except Exception as e:
    print(f"Database error: {e}")
finally:
    if cursor:
        cursor.close()
    if connection:
        connection.close()
```

## Performance Considerations
- **Large Tables**: `inventory_round_ed` (235M rows) - always filter by match_id or date range
- **Joins**: Use indexed columns (match_id, round_id) for optimal performance  
- **Aggregations**: Pre-calculate team totals rather than repeated player summations
- **Memory**: Process data in chunks for large datasets (>1M rounds)

## Output Standards
- **Statistical Significance**: Always report p-values for correlations
- **Effect Sizes**: Include coefficient magnitudes and confidence intervals
- **Practical Significance**: Relate statistical findings to CS:GO context
- **Reproducibility**: Document random seeds and data filtering criteria

The following is the current structure of the PostgreSQL database: 
================================================================================
DATABASE OVERVIEW - CS:GO Analytics
================================================================================

📊 TABLES (18 total):
----------------------------------------
• bomb_events_round_ed (BASE TABLE)
• hltv_events (BASE TABLE)
• hltv_events_prize_distribution (BASE TABLE)
• hltv_events_teams (BASE TABLE)
• hltv_match_info (BASE TABLE)
• hltv_match_map (BASE TABLE)
• inventory_round_ed (BASE TABLE)
• kills_round_ed (BASE TABLE)
• player_round_ed (BASE TABLE)
• rounds_ed (BASE TABLE)

🔍 TABLE: bomb_events_round_ed
------------------------------------------------------------
Columns (8 total):
  • id                        integer                NOT NULL DEFAULT nextval('bomb_events_round_ed_id_seq'::regclass)
  • round_id                  integer                NULL
  • tick                      bigint                NULL
  • bomb_event_type           integer                NULL
  • site                      text                NULL
  • carrier_hltv_id           bigint                NULL
  • carrier_hltv_name         text                NULL
  • inserted_at               timestamp with time zone                NULL DEFAULT now()

📈 Row count: 13,073,944

📋 Sample data (first 3 rows):
  Columns shown: id, round_id, tick, bomb_event_type, site...
  Row 1: ['36501', '5891', '2114', '3', 'None']
  Row 2: ['36502', '5891', '2189', '4', 'None']
  Row 3: ['36503', '5891', '4526', '3', 'None']



🔍 TABLE: hltv_events
------------------------------------------------------------
Columns (15 total):
  • auto_id                   integer                NOT NULL DEFAULT nextval('hltv_events_auto_id_seq'::regclass)
  • event_id                  integer                NOT NULL
  • name                      character varying(191)           NULL
  • date_start                timestamp with time zone                NULL
  • date_end                  timestamp with time zone                NULL
  • prize_pool                character varying(255)           NULL
  • location                  character varying(255)           NULL
  • number_teams              integer                NULL
  • event_infos_status        integer                NULL DEFAULT 0
  • date_added                timestamp with time zone                NULL DEFAULT CURRENT_TIMESTAMP
  • download_started          timestamp with time zone                NULL
  • download_completed        timestamp with time zone                NULL
  • temp_status               integer                NULL DEFAULT 0
  • temp_started              timestamp with time zone                NULL
  • event_pool                character varying(255)           NULL

📈 Row count: 6,530

📋 Sample data (first 3 rows):
  Columns shown: auto_id, event_id, name, date_start, date_end...
  Row 1: ['1', '2086', 'Rising Stars Balkan ...', '2015-12-18 11:00:00+...', '2015-12-20 11:00:00+...']
  Row 2: ['2', '2090', 'D!ngIT $2000 Weekly ...', '2015-12-21 11:00:00+...', '2015-12-22 11:00:00+...']
  Row 3: ['3', '2095', 'D!ngIT $2000 Weekly ...', '2016-01-04 11:00:00+...', '2016-01-05 11:00:00+...']


🔍 TABLE: hltv_events_prize_distribution
------------------------------------------------------------
Columns (9 total):
  • auto_id                   integer                NOT NULL DEFAULT nextval('hltv_events_prize_distribution_auto_id_seq'::regclass)
  • event_id                  integer                NOT NULL
  • team_name                 character varying(255)           NULL
  • team_id                   integer                NULL
  • place                     character varying(255)           NULL
  • prize                     character varying(255)           NULL
  • other_prize               character varying(255)           NULL
  • date_added                timestamp with time zone                NULL DEFAULT CURRENT_TIMESTAMP
  • status                    integer                NULL DEFAULT 0

📈 Row count: 43,215

📋 Sample data (first 3 rows):
  Columns shown: auto_id, event_id, team_name, team_id, place...
  Row 1: ['1', '5993', 'AaB', '9621', '1st']
  Row 2: ['2', '5993', 'Tricked', '4602', '2nd']
  Row 3: ['3', '5993', 'Quantum Bellator Fir...', '7367', '3-4th']


🔍 TABLE: hltv_events_teams
------------------------------------------------------------
Columns (8 total):
  • auto_id                   integer                NOT NULL DEFAULT nextval('hltv_events_teams_auto_id_seq'::regclass)
  • event_id                  integer                NOT NULL
  • team_name                 character varying(255)           NULL
  • team_id                   integer                NULL
  • rank_during               character varying(255)           NULL
  • date_added                timestamp with time zone                NULL DEFAULT CURRENT_TIMESTAMP
  • status                    integer                NULL DEFAULT 0
  • participation_reason      character varying(255)           NULL

📈 Row count: 50,905

📋 Sample data (first 3 rows):
  Columns shown: auto_id, event_id, team_name, team_id, rank_during...
  Row 1: ['1', '5993', 'Tricked', '4602', '92']
  Row 2: ['2', '5993', 'MASONIC', '10867', '139']
  Row 3: ['3', '5993', 'AaB', '9621', 'None']


🔍 TABLE: hltv_match_info
------------------------------------------------------------
Columns (19 total):
  • auto_id                   integer                NOT NULL DEFAULT nextval('hltv_match_info_auto_id_seq'::regclass)
  • match_id                  integer                NOT NULL
  • event_id                  integer                NULL
  • stats_id                  integer                NULL
  • significance              character varying(255)           NULL
  • date                      timestamp with time zone                NULL
  • format_type               character varying(255)           NULL
  • demo_link                 character varying(255)           NULL
  • team_1_name               character varying(255)           NULL
  • team_1_id                 integer                NULL
  • team_2_name               character varying(255)           NULL
  • team_2_id                 integer                NULL
  • winner_team_name          character varying(255)           NULL
  • winner_team_id            character varying(255)           NULL
  • maps_json                 character varying(2000)          NULL
  • players_json              character varying(2000)          NULL
  • veteos_json               character varying(2000)          NULL
  • info_status               integer                NULL DEFAULT 0
  • date_info_added           timestamp with time zone                NULL DEFAULT CURRENT_TIMESTAMP

📈 Row count: 78,486

📋 Sample data (first 3 rows):
  Columns shown: auto_id, match_id, event_id, stats_id, significance...
  Row 1: ['1', '2349876', '5993', '82578', 'Grand final']
  Row 2: ['2', '2349888', '5994', '82515', 'Grand final']
  Row 3: ['4', '2349874', '5993', '82559', 'Semi-final']


🔍 TABLE: hltv_match_map
------------------------------------------------------------
Columns (15 total):
  • auto_id                   integer                NOT NULL DEFAULT nextval('hltv_match_map_auto_id_seq'::regclass)
  • match_id                  integer                NOT NULL
  • map_name                  character varying(255)           NULL
  • stats_id                  integer                NULL
  • team_1_rounds             integer                NULL
  • team_2_rounds             integer                NULL
  • team_1_rounds_1           integer                NULL
  • team_2_rounds_1           integer                NULL
  • team_1_rounds_2           integer                NULL
  • team_2_rounds_2           integer                NULL
  • team_1_rounds_ot          integer                NULL
  • team_2_rounds_ot          integer                NULL
  • status                    integer                NULL DEFAULT 0
  • date_added                timestamp with time zone                NULL DEFAULT CURRENT_TIMESTAMP
  • t1_winner                 boolean                NULL DEFAULT false

📈 Row count: 154,391

📋 Sample data (first 3 rows):
  Columns shown: auto_id, match_id, map_name, stats_id, team_1_rounds...
  Row 1: ['1', '2349876', 'de_nuke', '123361', '12']
  Row 2: ['2', '2349876', 'de_inferno', '123366', '9']
  Row 3: ['3', '2349874', 'de_train', '123304', '17']


🔍 TABLE: inventory_round_ed
------------------------------------------------------------
Columns (8 total):
  • id                        integer                NOT NULL DEFAULT nextval('inventory_round_ed_id_seq'::regclass)
  • player_round_id           integer                NULL
  • round_id                  integer                NULL
  • event_type                integer                NULL
  • equipment_type            integer                NULL
  • equipment_string          text                NULL
  • equipment_class           integer                NULL
  • inserted_at               timestamp with time zone                NULL DEFAULT now()

📈 Row count: 235,171,866

📋 Sample data (first 3 rows):
  Columns shown: id, player_round_id, round_id, event_type, equipment_type...
  Row 1: ['227842097', '21307940', '2130820', '0', '2']
  Row 2: ['227842098', '21307940', '2130820', '0', '405']
  Row 3: ['227842099', '21307940', '2130820', '0', '404']


🔍 TABLE: kills_round_ed
------------------------------------------------------------
Columns (14 total):
  • id                        integer                NOT NULL DEFAULT nextval('kills_round_ed_id_seq'::regclass)
  • round_id                  integer                NULL
  • tick                      bigint                NULL
  • is_after_re               boolean                NULL
  • killer_hltv_id            bigint                NULL
  • killer_hltv_name          text                NULL
  • victim_hltv_id            bigint                NULL
  • victim_hltv_name          text                NULL
  • weapon                    integer                NULL
  • is_headshot               boolean                NULL
  • is_wallbang               boolean                NULL
  • assister_hltv_id          bigint                NULL
  • assister_hltv_name        text                NULL
  • inserted_at               timestamp with time zone                NULL DEFAULT now()

📈 Row count: 14,906,145

📋 Sample data (first 3 rows):
  Columns shown: id, round_id, tick, is_after_re, killer_hltv_id...
  Row 1: ['38860', '5891', '7662', 'False', '22965']
  Row 2: ['38861', '5891', '8306', 'False', '21100']
  Row 3: ['38862', '5891', '8561', 'False', '21100']



🔍 TABLE: rounds_ed
------------------------------------------------------------
Columns (14 total):
  • id                        integer                NOT NULL DEFAULT nextval('rounds_ed_id_seq'::regclass)
  • match_id                  bigint                NOT NULL
  • event_id                  bigint                NULL
  • id_demo_exports           integer                NOT NULL
  • map_number                integer                NULL
  • map_name                  text                NULL
  • round_num                 integer                NOT NULL
  • team1_winner              boolean                NULL
  • ct_winner                 boolean                NULL
  • is_ct_t1                  boolean                NULL
  • is_ot                     boolean                NULL
  • is_last_round_half        boolean                NULL
  • is_side_switch            boolean                NULL
  • inserted_at               timestamp with time zone                NULL DEFAULT now()

📈 Row count: 2,201,775

📋 Sample data (first 3 rows):
  Columns shown: id, match_id, event_id, id_demo_exports, map_number...
  Row 1: ['5891', '2366874', '7389', '184', '1']
  Row 2: ['5892', '2366874', '7389', '184', '1']
  Row 3: ['5893', '2366874', '7389', '184', '1']

🔗 FOREIGN KEY RELATIONSHIPS:
----------------------------------------
• No foreign key relationships found

================================================================================

