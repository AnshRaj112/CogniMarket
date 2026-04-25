import PyPDF2
with open("Before.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
with open("Before.txt", "w", encoding="utf-8") as out_f:
    out_f.write(text)
