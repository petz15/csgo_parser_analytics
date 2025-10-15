# CS:GO Analytics Database Schema Documentation
Generated: 2025-10-12 21:20:29

## Database Overview
- **Database**: csgo_parsed
- **Host**: 192.168.1.100:5444
- **Total Tables**: 17

## All Tables Summary
- **bomb_events_round_ed**: 15,473,392 rows
- **demo_exports**: 147,248 rows
- **demo_processing_status**: 114,515 rows
- **hltv_events**: 6,530 rows
- **hltv_events_prize_distribution**: 43,215 rows
- **hltv_events_teams**: 50,905 rows
- **hltv_match_info**: 78,486 rows
- **hltv_match_map**: 154,391 rows
- **inventory_round_ed**: 277,923,636 rows
- **kills_round_ed**: 17,584,321 rows
- **match_downloads**: 77,081 rows
- **parser_logs**: 0 rows
- **player_economy_ed**: 24,896,913 rows
- **player_round_ed**: 26,015,660 rows
- **round_streaks_ed**: 2,601,592 rows
- **rounds_ed**: 2,601,592 rows
- **team_economy_ed**: 4,979,508 rows

## Detailed Table Schemas

### rounds_ed (2,601,592 rows)

**Columns:**
- `id` - integer NOT NULL DEFAULT nextval('rounds_ed_id_seq'::regclass)
- `match_id` - bigint NOT NULL
- `event_id` - bigint NULL
- `id_demo_exports` - integer NOT NULL
- `map_number` - integer NULL
- `map_name` - text NULL
- `round_num` - integer NOT NULL
- `team1_winner` - boolean NULL
- `ct_winner` - boolean NULL
- `is_ct_t1` - boolean NULL
- `is_ot` - boolean NULL
- `is_last_round_half` - boolean NULL
- `is_side_switch` - boolean NULL
- `inserted_at` - timestamp with time zone NULL DEFAULT now()
- `re_tick` - integer NULL
- `rs_tick` - integer NULL
- `fte_tick` - integer NULL
- `rofe_tick` - integer NULL
- `t1_score_start` - integer NULL
- `t1_score_end` - integer NULL
- `t2_score_start` - integer NULL
- `t2_score_end` - integer NULL
- `status` - integer NULL
- `round_end_reason` - integer NULL

**Sample Data:**
```
Columns: id, match_id, event_id, id_demo_exports, map_number
Row 1: ['24794', '2366884', '7377', '570', '1']
Row 2: ['24804', '2366884', '7377', '570', '1']
```

### player_round_ed (26,015,660 rows)

**Columns:**
- `id` - integer NOT NULL DEFAULT nextval('player_round_ed_id_seq'::regclass)
- `player_id` - bigint NULL
- `round_id` - integer NULL
- `is_alive_fte` - boolean NULL
- `is_alive_re` - boolean NULL
- `is_alive_rofe` - boolean NULL
- `kills` - integer NULL
- `assists` - integer NULL
- `deaths` - integer NULL
- `has_defuse_kit_fte` - boolean NULL
- `has_defuse_kit_re` - boolean NULL
- `has_helmet_fte` - boolean NULL
- `health` - integer NULL
- `armor` - integer NULL
- `money_rs` - integer NULL
- `money_fte` - integer NULL
- `money_re` - integer NULL
- `money_rofe` - integer NULL
- `eq_val_rs` - integer NULL
- `eq_val_fte` - integer NULL
- `eq_val_re` - integer NULL
- `eq_val_rofe` - integer NULL
- `inserted_at` - timestamp with time zone NULL DEFAULT now()
- `team` - integer NULL
- `armor_fte` - integer NULL

**Sample Data:**
```
Columns: id, player_id, round_id, is_alive_fte, is_alive_re
Row 1: ['18468195', '13296', '1846843', 'True', 'False']
Row 2: ['18468199', '13361', '1846843', 'True', 'True']
```

### kills_round_ed (17,584,321 rows)

**Columns:**
- `id` - integer NOT NULL DEFAULT nextval('kills_round_ed_id_seq'::regclass)
- `round_id` - integer NULL
- `tick` - bigint NULL
- `is_after_re` - boolean NULL
- `killer_hltv_id` - bigint NULL
- `killer_hltv_name` - text NULL
- `victim_hltv_id` - bigint NULL
- `victim_hltv_name` - text NULL
- `weapon` - integer NULL
- `is_headshot` - boolean NULL
- `is_wallbang` - boolean NULL
- `assister_hltv_id` - bigint NULL
- `assister_hltv_name` - text NULL
- `inserted_at` - timestamp with time zone NULL DEFAULT now()

**Sample Data:**
```
Columns: id, round_id, tick, is_after_re, killer_hltv_id
Row 1: ['38860', '5891', '7662', 'False', '22965']
Row 2: ['38861', '5891', '8306', 'False', '21100']
```

### bomb_events_round_ed (15,473,392 rows)

**Columns:**
- `id` - integer NOT NULL DEFAULT nextval('bomb_events_round_ed_id_seq'::regclass)
- `round_id` - integer NULL
- `tick` - bigint NULL
- `bomb_event_type` - integer NULL
- `site` - text NULL
- `carrier_hltv_id` - bigint NULL
- `carrier_hltv_name` - text NULL
- `inserted_at` - timestamp with time zone NULL DEFAULT now()
- `after_re` - boolean NULL

**Sample Data:**
```
Columns: id, round_id, tick, bomb_event_type, site
Row 1: ['4446953', '721521', '66701', '1', 'B']
Row 2: ['4446954', '721521', '67352', '4', 'None']
```

### inventory_round_ed (277,923,636 rows)

**Columns:**
- `id` - integer NOT NULL DEFAULT nextval('inventory_round_ed_id_seq'::regclass)
- `player_round_id` - integer NULL
- `round_id` - integer NULL
- `event_type` - integer NULL
- `equipment_type` - integer NULL
- `equipment_string` - text NULL
- `equipment_class` - integer NULL
- `inserted_at` - timestamp with time zone NULL DEFAULT now()

**Sample Data:**
```
Columns: id, player_round_id, round_id, event_type, equipment_type
Row 1: ['641548', '58874', '5891', '0', '2']
Row 2: ['641549', '58874', '5891', '0', '404']
```

### team_economy_ed (4,979,508 rows)

**Columns:**
- `id` - integer NOT NULL DEFAULT nextval('team_economy_ed_id_seq'::regclass)
- `round_id` - integer NULL
- `team` - integer NOT NULL
- `money_earned` - integer NULL
- `total_donations` - integer NULL
- `calculated_at` - timestamp with time zone NULL DEFAULT now()

**Sample Data:**
```
Columns: id, round_id, team, money_earned, total_donations
Row 1: ['6588896', '9602', '1', '3250', '-3300']
Row 2: ['6588897', '9614', '1', '2700', '-7350']
```

### player_economy_ed (24,896,913 rows)

**Columns:**
- `id` - integer NOT NULL DEFAULT nextval('player_economy_ed_id_seq'::regclass)
- `player_round_id` - integer NULL
- `round_id` - integer NULL
- `player_id` - bigint NULL
- `money_earned` - integer NULL
- `money_earned_re` - integer NULL
- `money_spent` - integer NULL
- `saved_eq_val` - integer NULL
- `starting_eq_val` - integer NULL
- `donations` - integer NULL
- `calculation_case` - integer NULL
- `notes` - text NULL
- `calculated_at` - timestamp with time zone NULL DEFAULT now()
- `calculation_case_next` - integer NULL

**Sample Data:**
```
Columns: id, player_round_id, round_id, player_id, money_earned
Row 1: ['32929592', '95524', '9557', '14390', '1900']
Row 2: ['32929593', '95534', '9557', '14394', '1900']
```

## Foreign Key Relationships

No formal foreign key constraints found.

## Common Query Patterns

### Basic Round Analysis
```sql
SELECT 
    r.id,
    r.match_id,
    r.round_num,
    r.team1_winner,
    r.ct_winner
FROM rounds_ed r
WHERE r.match_id = YOUR_MATCH_ID
ORDER BY r.round_num;
```

### Player Performance
```sql
SELECT 
    p.player_hltv_name,
    p.team_num,
    AVG(p.kills) as avg_kills,
    AVG(p.deaths) as avg_deaths,
    AVG(p.money_spent) as avg_spending
FROM player_round_ed p
JOIN rounds_ed r ON p.round_id = r.id
WHERE r.match_id = YOUR_MATCH_ID
GROUP BY p.player_hltv_name, p.team_num;
```
