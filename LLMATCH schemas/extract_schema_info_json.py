
import os
import json

def extract_schema_info_json():
    input_file = 'sakila-schema.json' #replace
    output_dir = 'schema_documentations'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    table_count = 0
    column_count = 0

    for table_key, table_data in schema.items():
        table_name = table_data.get('table', table_key).lower()
        table_description = table_data.get('table_description', '').strip()

        # Write table-level description
        table_filename = f"sakila-table-{table_name}.txt" #replace
        table_filepath = os.path.join(output_dir, table_filename)
        with open(table_filepath, 'w', encoding='utf-8') as f:
            f.write(table_description)
        print(f"Created {table_filename}")
        table_count += 1

        # Write each column description
        columns = table_data.get('columns', {})
        for col_key, col_data in columns.items():
            col_name = col_data.get('name', col_key).lower()
            col_description = col_data.get('description', '').strip()

            col_filename = f"sakila-{table_name}-{col_name}.txt" #replace
            col_filepath = os.path.join(output_dir, col_filename)
            with open(col_filepath, 'w', encoding='utf-8') as f:
                f.write(col_description)
            print(f"  Created {col_filename}")
            column_count += 1

    print(f"\nTotal tables extracted : {table_count}")
    print(f"Total columns extracted: {column_count}")

if __name__ == "__main__":
    extract_schema_info_json()
