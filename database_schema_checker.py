#!/usr/bin/env python3
"""
Database Schema Verification Script
Checks the actual database structure and generates accurate documentation
"""

import psycopg2
import pandas as pd
from datetime import datetime

# Database configuration
DB_CONFIG = {
    "dbname": "csgo_parsed",
    "user": "csgo_parser",
    "password": "3?6B7yTGPrkJF34p",
    "host": "192.168.1.100",
    "port": "5444"
}

def get_database_connection():
    """Establish database connection"""
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        print("✅ Database connection established")
        return connection
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None

def get_all_tables(connection):
    """Get all tables in the database with actual row counts"""
    # First get all table names
    query = """
    SELECT table_schema, table_name
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE'
    ORDER BY table_name;
    """
    
    cursor = connection.cursor()
    cursor.execute(query)
    tables = cursor.fetchall()
    
    # Get actual row count for each table
    table_info = []
    for schema, table_name in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            table_info.append((schema, table_name, row_count))
            print(f"   • {table_name}: {row_count:,} rows")
        except Exception as e:
            print(f"   ⚠️  Error counting {table_name}: {e}")
            table_info.append((schema, table_name, 0))
    
    cursor.close()
    return table_info

def get_table_structure(connection, table_name):
    """Get detailed structure of a specific table"""
    query = """
    SELECT 
        column_name,
        data_type,
        character_maximum_length,
        is_nullable,
        column_default,
        ordinal_position
    FROM information_schema.columns 
    WHERE table_name = %s 
    AND table_schema = 'public'
    ORDER BY ordinal_position;
    """
    
    cursor = connection.cursor()
    cursor.execute(query, (table_name,))
    columns = cursor.fetchall()
    cursor.close()
    
    return columns

def get_sample_data(connection, table_name, limit=3):
    """Get sample data from table"""
    try:
        cursor = connection.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        cursor.close()
        return data, column_names
    except Exception as e:
        print(f"Error getting sample data from {table_name}: {e}")
        return [], []

def get_foreign_keys(connection):
    """Get all foreign key relationships"""
    query = """
    SELECT
        tc.table_name, 
        kcu.column_name, 
        ccu.table_name AS foreign_table_name,
        ccu.column_name AS foreign_column_name 
    FROM 
        information_schema.table_constraints AS tc 
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
          AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
          AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY' 
    AND tc.table_schema = 'public';
    """
    
    cursor = connection.cursor()
    cursor.execute(query)
    fks = cursor.fetchall()
    cursor.close()
    
    return fks

def analyze_key_tables(connection):
    """Analyze the most important tables in detail"""
    key_tables = [
        'rounds_ed',
        'player_round_ed', 
        'kills_round_ed',
        'bomb_events_round_ed',
        'inventory_round_ed',
        'team_economy_ed',
        'player_economy_ed'
    ]
    
    analysis = {}
    
    for table in key_tables:
        print(f"\n🔍 Analyzing {table}...")
        
        # Get structure
        columns = get_table_structure(connection, table)
        if not columns:
            print(f"   ⚠️  Table {table} not found or no columns")
            continue
            
        # Get sample data
        sample_data, col_names = get_sample_data(connection, table, 2)
        
        # Get row count
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
        except:
            row_count = 0
        cursor.close()
        
        analysis[table] = {
            'columns': columns,
            'sample_data': sample_data,
            'column_names': col_names,
            'row_count': row_count
        }
        
        print(f"   ✅ {len(columns)} columns, {row_count:,} rows")
    
    return analysis

def generate_documentation(tables_info, key_analysis, foreign_keys):
    """Generate accurate documentation content"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    doc = f"""# CS:GO Analytics Database Schema Documentation
Generated: {timestamp}

## Database Overview
- **Database**: csgo_parsed
- **Host**: 192.168.1.100:5444
- **Total Tables**: {len(tables_info)}

## All Tables Summary
"""
    
    for schema, table_name, row_count in tables_info:
        doc += f"- **{table_name}**: {row_count:,} rows\n"
    
    doc += "\n## Detailed Table Schemas\n\n"
    
    # Document key tables in detail
    for table_name, analysis in key_analysis.items():
        if not analysis['columns']:
            continue
            
        doc += f"### {table_name} ({analysis['row_count']:,} rows)\n\n"
        
        doc += "**Columns:**\n"
        for col_name, data_type, max_len, nullable, default, pos in analysis['columns']:
            type_info = data_type
            if max_len:
                type_info += f"({max_len})"
            nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
            default_str = f" DEFAULT {default}" if default else ""
            doc += f"- `{col_name}` - {type_info} {nullable_str}{default_str}\n"
        
        # Add sample data if available
        if analysis['sample_data'] and analysis['column_names']:
            doc += f"\n**Sample Data:**\n"
            doc += "```\n"
            # Show first few columns to avoid wide output
            display_cols = analysis['column_names'][:5]
            doc += f"Columns: {', '.join(display_cols)}\n"
            
            for i, row in enumerate(analysis['sample_data'][:2], 1):
                display_values = [str(val)[:30] + '...' if len(str(val)) > 30 else str(val) 
                                for val in row[:5]]
                doc += f"Row {i}: {display_values}\n"
            doc += "```\n"
        
        doc += "\n"
    
    # Add foreign key relationships
    doc += "## Foreign Key Relationships\n\n"
    if foreign_keys:
        for table, column, ref_table, ref_column in foreign_keys:
            doc += f"- `{table}.{column}` → `{ref_table}.{ref_column}`\n"
    else:
        doc += "No formal foreign key constraints found.\n"
    
    doc += "\n## Common Query Patterns\n\n"
    doc += """### Basic Round Analysis
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
"""
    
    return doc

def main():
    """Main execution function"""
    print("🔍 CS:GO Database Schema Verification")
    print("=" * 50)
    
    # Connect to database
    connection = get_database_connection()
    if not connection:
        return
    
    try:
        # Get all tables
        print("\n📊 Getting all tables...")
        tables_info = get_all_tables(connection)
        print(f"Found {len(tables_info)} tables")
        
        # Analyze key tables
        print("\n🔍 Analyzing key tables in detail...")
        key_analysis = analyze_key_tables(connection)
        
        # Get foreign keys
        print("\n🔗 Checking foreign key relationships...")
        foreign_keys = get_foreign_keys(connection)
        print(f"Found {len(foreign_keys)} foreign key relationships")
        
        # Generate documentation
        print("\n📝 Generating documentation...")
        documentation = generate_documentation(tables_info, key_analysis, foreign_keys)
        
        # Save to file
        with open('database_schema_verified.md', 'w', encoding='utf-8') as f:
            f.write(documentation)
        
        print("✅ Documentation saved to 'database_schema_verified.md'")
        
        # Print summary
        print(f"\n📋 SUMMARY:")
        print(f"   • Total tables: {len(tables_info)}")
        print(f"   • Key tables analyzed: {len(key_analysis)}")
        print(f"   • Foreign keys: {len(foreign_keys)}")
        
        # Show largest tables
        sorted_tables = sorted(tables_info, key=lambda x: x[2] or 0, reverse=True)
        print(f"\n📈 Largest tables:")
        for schema, table, rows in sorted_tables[:5]:
            if rows:
                print(f"   • {table}: {rows:,} rows")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
    
    finally:
        connection.close()
        print("\n🔚 Database connection closed")

if __name__ == "__main__":
    main()