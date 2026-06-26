import markdown
import os

md_file = "/home/exouser/pruning/custom/llama_6hop/final_logic_heuristics_report.md"
html_file = "/home/exouser/pruning/custom/llama_6hop/final_logic_heuristics_report.html"

with open(md_file, "r") as f:
    text = f.read()

# Convert markdown to html, enabling tables extension
html_body = markdown.markdown(text, extensions=['tables'])

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logic Heuristics Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #f8f9fa;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            margin-top: 2rem;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background-color: #f1f8ff;
            color: #2c3e50;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        code {{
            background-color: #e8e8e8;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: Consolas, monospace;
            font-size: 0.9em;
        }}
        .container {{
            background-color: white;
            padding: 3rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>
"""

with open(html_file, "w") as f:
    f.write(html_template)

print(f"Successfully created {html_file}")
