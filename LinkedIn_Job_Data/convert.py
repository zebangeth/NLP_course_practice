from bs4 import BeautifulSoup
import pandas as pd

def html_to_csv(html_file, output_csv):
    # Read the HTML file
    with open(html_file, 'r') as f:
        html_content = f.read()

    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Initialize lists to store data
    times = []
    steps = []
    losses = []
    
    # Find all table rows (excluding header)
    rows = soup.find_all('tr', class_='css-1792h1')
    
    # Extract data from each row
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 3:  # Ensure row has enough columns
            times.append(cols[0].text.strip())
            steps.append(int(cols[1].text.strip()))
            losses.append(float(cols[2].text.strip()))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time': times,
        'Step': steps,
        'Training_Loss': losses
    })
    
    # Sort by step number in descending order (to match original data)
    df = df.sort_values('Step', ascending=False)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

# Usage
html_file = 'a.html'
output_csv = 'adata.csv'
html_to_csv(html_file, output_csv)