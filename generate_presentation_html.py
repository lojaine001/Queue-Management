"""
Convert presentation_rex_poc.md to a beautifully styled, interactive HTML presentation.
Run from the Queue-Management directory:
    python generate_presentation_html.py
"""

import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_MD = os.path.join(SCRIPT_DIR, "presentation_rex_poc.md")
OUTPUT_HTML = os.path.join(SCRIPT_DIR, "presentation_rex_poc.html")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>REX POC — IQMS Presentation</title>
    <style>
        :root {{
            --primary: #0066cc;
            --background: #f5f9fd;
            --slide-bg: linear-gradient(135deg, #f5f9fd 0%, #e6f2fc 100%);
            --text: #1d1d1f;
            --text-secondary: #6e6e73;
            --border: #d2d2d7;
            --code-bg: rgba(0, 113, 227, 0.05);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--background);
            color: var(--text);
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }}

        /* Slide Deck Container */
        .deck-container {{
            width: 100vw;
            height: 100vh;
            position: relative;
            background: var(--slide-bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(225, 238, 253, 0.6) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(200, 225, 252, 0.6) 0px, transparent 50%),
                var(--slide-bg);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        /* Slide Viewport */
        .slide-viewport {{
            flex: 1;
            position: relative;
            padding: 50px 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        /* Slide Content Rules */
        .slide {{
            display: none;
            width: 100%;
            height: 100%;
            opacity: 0;
            transition: opacity 0.3s ease-in-out;
            animation: fadeIn 0.3s forwards;
        }}

        .slide.active {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            opacity: 1;
        }}

        @keyframes fadeIn {{
            to {{ opacity: 1; }}
        }}

        /* Typography */
        h1 {{
            font-size: 3.2rem;
            color: var(--primary);
            margin-bottom: 25px;
            font-weight: 700;
            line-height: 1.2;
        }}

        h2 {{
            font-size: 2.4rem;
            color: var(--text);
            margin-bottom: 20px;
            font-weight: 700;
        }}

        p {{
            font-size: 1.45rem;
            line-height: 1.7;
            margin-bottom: 15px;
        }}

        ul {{
            margin-left: 30px;
            margin-bottom: 20px;
        }}

        li {{
            font-size: 1.45rem;
            line-height: 1.7;
            margin-bottom: 12px;
        }}

        strong {{
            font-weight: 600;
        }}

        code {{
            font-family: "Courier New", Courier, monospace;
            background-color: var(--code-bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
            border: 1px solid var(--border);
        }}

        pre {{
            background-color: var(--code-bg);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--border);
            overflow-x: auto;
            margin-bottom: 15px;
        }}

        pre code {{
            background: none;
            border: none;
            padding: 0;
            font-size: 0.85em;
            line-height: 1.4;
        }}

        /* Cover Slide Specifics */
        .cover-slide {{
            text-align: left;
            align-items: stretch;
            justify-content: center;
        }}

        .cover-slide h1 {{
            font-size: 2.8rem;
            margin-bottom: 15px;
            color: var(--primary);
        }}

        .cover-slide h2 {{
            font-size: 1.5rem;
            color: var(--text-secondary);
            font-weight: 400;
            margin-bottom: 25px;
        }}

        .cover-slide p {{
            font-size: 1.1rem;
            color: var(--text-secondary);
        }}

        /* Two Column Layout Grid */
        .two-col {{
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 50px;
            align-items: center;
            width: 100%;
            margin-top: 10px;
            flex: 1;
        }}

        .left-col {{
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .right-col {{
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        /* Image Display */
        .slide-image-container {{
            display: flex;
            justify-content: center;
            width: 100%;
        }}

        .slide-image-container img {{
            max-width: 100%;
            max-height: 520px;
            object-fit: contain;
            border-radius: 8px;
            border: 1px solid var(--border);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        }}

        .cover-slide .slide-image-container img {{
            max-height: 520px;
            max-width: 100%;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            margin-bottom: 15px;
        }}

        th, td {{
            padding: 10px 14px;
            text-align: left;
            font-size: 0.95rem;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background-color: var(--primary);
            color: white;
            font-weight: 600;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}

        tr:nth-child(even) td {{
            background-color: var(--code-bg);
        }}

        /* Footer Controls */
        .footer {{
            height: 48px;
            border-top: 1px solid #e5e5e7;
            padding: 0 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #fafafa;
        }}

        .footer-title {{
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text);
        }}

        .footer-title span {{
            color: var(--text-secondary);
            font-weight: 400;
            margin-left: 10px;
        }}

        .controls {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .control-btn {{
            background: none;
            border: none;
            cursor: pointer;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--primary);
            transition: background-color 0.2s;
        }}

        .control-btn:hover {{
            background-color: #e8f2fc;
        }}

        .control-btn:disabled {{
            color: var(--text-secondary);
            cursor: not-allowed;
            background: none;
        }}

        .counter {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        /* Progress Bar */
        .progress-bar-container {{
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background-color: #e5e5e7;
        }}

        .progress-bar {{
            height: 100%;
            width: 0%;
            background-color: var(--primary);
            transition: width 0.3s ease;
        }}

        /* Keyboard Tips overlay */
        .tips {{
            position: absolute;
            bottom: 60px;
            right: 30px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.5s;
        }}

        .deck-container:hover .tips {{
            opacity: 1;
        }}
    </style>
</head>
<body>

    <div class="deck-container">
        <div class="slide-viewport">
            {slides_html}
            <div class="tips">Utilisez les touches ◄ / ► ou Espace pour naviguer</div>
        </div>

        <div class="footer">
            <div class="footer-title">
                REX POC — Système Intelligent de Gestion de File d'Attente (IQMS) <span>| Synthèse d'Intégration & REX</span>
            </div>
            <div class="controls">
                <button id="prev-btn" class="control-btn" onclick="prevSlide()">Précédent</button>
                <div class="counter" id="counter">1 / X</div>
                <button id="next-btn" class="control-btn" onclick="nextSlide()">Suivant</button>
            </div>
        </div>

        <div class="progress-bar-container">
            <div class="progress-bar" id="progress"></div>
        </div>
    </div>

    <script>
        let currentSlideIdx = 0;
        const slides = document.querySelectorAll('.slide');
        const counter = document.getElementById('counter');
        const progressBar = document.getElementById('progress');
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');

        function updateDeck() {{
            slides.forEach((slide, idx) => {{
                if (idx === currentSlideIdx) {{
                    slide.classList.add('active');
                }} else {{
                    slide.classList.remove('active');
                }}
            }});

            counter.innerText = `${{currentSlideIdx + 1}} / ${{slides.length}}`;
            
            const pct = ((currentSlideIdx + 1) / slides.length) * 100;
            progressBar.style.width = `${{pct}}%`;

            prevBtn.disabled = currentSlideIdx === 0;
            nextBtn.disabled = currentSlideIdx === slides.length - 1;
        }}

        function nextSlide() {{
            if (currentSlideIdx < slides.length - 1) {{
                currentSlideIdx++;
                updateDeck();
            }}
        }}

        function prevSlide() {{
            if (currentSlideIdx > 0) {{
                currentSlideIdx--;
                updateDeck();
            }}
        }}

        // Keyboard Navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'Enter') {{
                nextSlide();
            }} else if (e.key === 'ArrowLeft' || e.key === 'Backspace') {{
                prevSlide();
            }}
        }});

        // Initialize presentation
        updateDeck();
    </script>
</body>
</html>
"""

def parse_markdown_to_html_slides(md_text):
    slides_raw = re.split(r'\n---\n|\r\n---\r\n', md_text)
    slides_html = []
    
    for slide_idx, slide_content in enumerate(slides_raw):
        slide_content = slide_content.strip()
        if not slide_content:
            continue
            
        # Skip Marp Frontmatter block if detected
        if "marp:" in slide_content and "theme:" in slide_content:
            continue
            
        lines = slide_content.splitlines()
        
        # Check if this slide is the cover slide
        is_cover = False
        if slide_idx == 0 or (len(lines) > 0 and lines[0].startswith("# ") and any("Restitution" in l or "Rapport" in l for l in lines)):
            is_cover = True

        def inline_html(text):
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
            text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
            text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
            text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
            return text

        # Pre-parse elements to determine if we should use a two-column layout
        has_image = False
        image_line = None
        has_text_or_list = False
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.startswith("!["):
                has_image = True
                image_line = line_stripped
            elif (line_stripped.startswith("* ") or 
                  line_stripped.startswith("- ") or 
                  (len(line_stripped) > 0 and not line_stripped.startswith("|") and not line_stripped.startswith("```"))):
                # Exclude title lines from two-column text detection
                if not line_stripped.startswith("# ") and not line_stripped.startswith("## "):
                    has_text_or_list = True

        use_two_col = has_image and has_text_or_list
        
        slide_body = []
        slide_body.append(f'<div class="slide{" cover-slide" if is_cover else ""}">')
        
        # Extract title first to display it outside the two-column grid
        slide_title = ""
        content_lines = []
        for line in lines:
            line_stripped = line.strip()
            # If it is the cover slide, keep the titles inside to render in the left column
            if (line_stripped.startswith("# ") or line_stripped.startswith("## ")) and not is_cover:
                m_h = re.match(r"^(#{1,4})\s+(.*)", line)
                if m_h:
                    tag = "h2"
                    slide_title = f"<{tag}>{inline_html(m_h.group(2))}</{tag}>"
            else:
                content_lines.append(line)

        if slide_title:
            slide_body.append(slide_title)

        if use_two_col:
            slide_body.append('<div class="two-col">')
            slide_body.append('<div class="left-col">')
            
        in_code = False
        in_table = False
        code_lines = []
        table_rows = []
        in_list = False
        
        def flush_code():
            nonlocal code_lines
            if code_lines:
                code_text = "\n".join(code_lines)
                slide_body.append(f"<pre><code>{inline_html(code_text)}</code></pre>")
                code_lines = []
                
        def flush_table():
            nonlocal table_rows
            if not table_rows:
                return
            html = ["<table>"]
            for ri, row in enumerate(table_rows):
                html.append("<tr>")
                tag = "th" if ri == 0 else "td"
                for cell in row:
                    html.append(f"<{tag}>{inline_html(cell.strip('`'))}</{tag}>")
                html.append("</tr>")
            html.append("</table>")
            slide_body.append("".join(html))
            table_rows = []

        # Parse content lines
        for line in content_lines:
            line_stripped = line.strip()
            
            # Fenced code block
            if line_stripped.startswith("```"):
                if not in_code:
                    in_code = True
                    code_lines = []
                else:
                    in_code = False
                    flush_code()
                continue
                
            if in_code:
                code_lines.append(line)
                continue
                
            # Table row
            if line_stripped.startswith("|"):
                if re.match(r"^[\|\s\-:]+$", line_stripped):
                    continue
                cells = [c.strip() for c in line_stripped.strip("|").split("|")]
                table_rows.append(cells)
                continue
            else:
                if table_rows:
                    flush_table()
                    
            # Headings in content
            m_h = re.match(r"^(#{1,4})\s+(.*)", line)
            if m_h:
                if in_list:
                    slide_body.append("</ul>")
                    in_list = False
                level = len(m_h.group(1))
                if is_cover:
                    tag = "h1" if level == 1 else "h2"
                else:
                    tag = "h3"
                text = inline_html(m_h.group(2))
                slide_body.append(f"<{tag}>{text}</{tag}>")
                continue
                
            # Bullets
            if line_stripped.startswith("* ") or line_stripped.startswith("- "):
                if not in_list:
                    slide_body.append("<ul>")
                    in_list = True
                text = inline_html(line_stripped[2:])
                slide_body.append(f"<li>{text}</li>")
                continue
                
            # Images
            m_img = re.match(r"^!\[(.*?)\]\((.*?)\)", line_stripped)
            if m_img:
                if in_list:
                    slide_body.append("</ul>")
                    in_list = False
                
                # If we are using two-column layout, we skip drawing it in the left text column
                if use_two_col:
                    continue
                    
                img_path = m_img.group(2)
                img_name = os.path.basename(img_path) if img_path.startswith("/") else img_path
                slide_body.append(f'<div class="slide-image-container"><img src="{img_name}" alt="{inline_html(m_img.group(1))}"></div>')
                continue
                
            # Paragraph text
            if line_stripped:
                if in_list:
                    slide_body.append("</ul>")
                    in_list = False
                slide_body.append(f"<p>{inline_html(line_stripped)}</p>")
                
        if in_list:
            slide_body.append("</ul>")
        if table_rows:
            flush_table()
        if code_lines:
            flush_code()
            
        if use_two_col:
            slide_body.append('</div>') # Close left-col
            slide_body.append('<div class="right-col">')
            
            # Draw the image inside the right column
            m_img = re.match(r"^!\[(.*?)\]\((.*?)\)", image_line)
            if m_img:
                img_path = m_img.group(2)
                img_name = os.path.basename(img_path) if img_path.startswith("/") else img_path
                slide_body.append(f'<div class="slide-image-container"><img src="{img_name}" alt="{inline_html(m_img.group(1))}"></div>')
                
            slide_body.append('</div>') # Close right-col
            slide_body.append('</div>') # Close two-col
            
        slide_body.append('</div>')
        slides_html.append("\n".join(slide_body))
        
    return slides_html

def main():
    if not os.path.exists(INPUT_MD):
        print(f"Error: Input Markdown file '{INPUT_MD}' not found.")
        sys.exit(1)
        
    print(f"Reading markdown presentation from '{INPUT_MD}'...")
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    slides = parse_markdown_to_html_slides(md_text)
    
    print(f"Generating interactive HTML to '{OUTPUT_HTML}'...")
    slides_joined = "\n".join(slides)
    html_output = HTML_TEMPLATE.format(slides_html=slides_joined)
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_output)
        
    print("Presentation HTML successfully generated.")

if __name__ == "__main__":
    main()
