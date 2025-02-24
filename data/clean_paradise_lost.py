import re
def clean_paradise_lost(input_path, output_path):
    # Read the input file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned = []
    start_processing = False
    
    for line in lines:
        # Strip whitespace and normalize line endings
        line = line.strip()
        
        # Check for the start marker
        if "*** START OF THE PROJECT GUTENBERG EBOOK PARADISE LOST ***" in line:
            start_processing = True
            continue  # Skip the marker line itself
            
        if not start_processing:
            continue
            
        # Skip empty lines
        if not line:
            continue
            
        # Normalize book headers to uppercase
        if re.match(r'^Book [IVXLCDM]+$', line, re.IGNORECASE):
            book_num = re.search(r'[IVXLCDM]+', line, re.IGNORECASE).group()
            line = f"BOOK {book_num}"
            
        cleaned.append(line + '\n')
    
    # Write cleaned content to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned)
        print(f"Cleaned file saved to {output_path}")

if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "paradise_lost_cleaned.txt"
    clean_paradise_lost(input_file, output_file)