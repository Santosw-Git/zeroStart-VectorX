# import PyPDF2
# import tiktoken

# def load_pdf(path):
#     text = ""
#     with open(path, "rb") as f:  
#         reader = PyPDF2.PdfReader(f)
#         for page in reader.pages:
#             text += page.extract_text()
#     return text

# pdf_text = load_pdf("/content/harrypotter.pdf")

# print(len(pdf_text))
# print(pdf_text[:500])

# tokenizer = tiktoken.get_encoding("gpt2")

# total_characters = len(pdf_text)
# total_tokens = len(tokenizer.encode(pdf_text))

# print("Characters:", total_characters)
# print("Tokens:", total_tokens)
# print("Sample tokens:", tokenizer.encode(pdf_text[:100]))

with open("the-verdict.txt", "r",encoding="utf-8") as f:
  pdf_text = f.read()   
print("Length of text:", len(pdf_text))
print("Sample text:\n", pdf_text[:500])