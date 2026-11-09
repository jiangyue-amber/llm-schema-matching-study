
import os
import re

def extract_schema_info():
    input_file = 'bank2-comment.sql'
    output_dir = 'schema_documentations'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex patterns — IGNORECASE handles 'IS' vs 'is'; DOTALL handles multi-line descriptions
    table_pattern = re.compile(r"COMMENT ON TABLE (\w+) IS '(.*?)';", re.IGNORECASE | re.DOTALL)
    column_pattern = re.compile(r"COMMENT ON COLUMN (\w+)\.(\w+) IS '(.*?)';", re.IGNORECASE | re.DOTALL)

    # Extract tables
    tables = table_pattern.findall(content)
    seen_tables = set()
    for table_name, description in tables:
        key = table_name.lower()
        if key in seen_tables:
            print(f"WARNING: duplicate table definition for '{table_name}', skipping.")
            continue
        seen_tables.add(key)
        filename = f"bank2-table-{key}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(description)
        print(f"Created {filename}")

    print(f"\nTotal tables extracted: {len(seen_tables)}")

    # Extract columns (attributes)
    columns = column_pattern.findall(content)
    for table_name, column_name, description in columns:
        filename = f"bank2-{table_name.lower()}-{column_name.lower()}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(description)
        print(f"Created {filename}")

    print(f"\nTotal columns extracted: {len(columns)}")

if __name__ == "__main__":
    extract_schema_info()
