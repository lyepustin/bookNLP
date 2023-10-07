from PIL import Image
import pytesseract
import pathlib

path_parent = pathlib.Path(__file__).parent.parent.resolve()
image = Image.open(f"{path_parent}/docs/app.jpg")
text = pytesseract.image_to_string(image)
print(text)
