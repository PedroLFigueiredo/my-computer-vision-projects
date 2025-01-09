import cv2
def create_sketch(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_invert = cv2.bitwise_not(imagem_cinza)
    imagem_blur = cv2.GaussianBlur(imagem_invert, (41, 41), sigmaX=0, sigmaY=0)
    imagem_blur_invert = cv2.bitwise_not(imagem_blur)
    imagem_sketch = cv2.divide(imagem_cinza, imagem_blur_invert, scale=256.0)
    return imagem_sketch

def load_image(filepath):
    imagem = cv2.imread(filepath)
    return imagem

def display_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    filepath = input("Filepath: ")
    imagem = load_image(filepath)
    if imagem is None:
        print("Error")
        return
    image_sketch = create_sketch(imagem)
    display_image(image_sketch)

if __name__ == "__main__":
    main()