from PIL import Image
import pytesseract
import pathlib

path_parent = pathlib.Path(__file__).parent.parent.resolve()
image = Image.open(f"/Users/denlyep/Desktop/a.png")
text = pytesseract.image_to_string(image)

with open('/Users/denlyep/Desktop/question.txt', 'w') as file:
    file.write(text)

print("done")