from PIL import Image

img = Image.open("/home/wasi/Downloads/unsplash wallpapers/test (6).png")
img = img.resize((28, 28), Image.ANTIALIAS)
img = img.convert('LA')
img.save('/home/wasi/Downloads/unsplash wallpapers/greyscale.png')
print(img)
img.show()
print("img downloaded successfully")