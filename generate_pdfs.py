"""
File to test where does ZhipuAI fail by generating increasingly large files
"""
import subprocess

for i in range(1, 101):
    filename = f"pdfs/pages_{i}.typ"
    with open(filename, 'x') as file:
        file.write("#lorem(680)")
        for j in range(1, i):
            file.write(f"\n#pagebreak()")
            file.write(f"\n#lorem(680)")
    
    # then run the following shell command
    # typst compile pdfs/pages_{i}.typ
    subprocess.run(["typst", "compile", filename])
