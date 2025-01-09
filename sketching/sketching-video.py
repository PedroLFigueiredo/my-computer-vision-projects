import cv2

def create_sketch(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_invert = cv2.bitwise_not(imagem_cinza)
    imagem_blur = cv2.GaussianBlur(imagem_invert, (41, 41), sigmaX=0, sigmaY=0)
    imagem_blur_invert = cv2.bitwise_not(imagem_blur)
    imagem_sketch = cv2.divide(imagem_cinza, imagem_blur_invert, scale=256.0)
    return imagem_sketch

def start_video(video):
    while True:
        ok, frame = video.read()
        if not ok:
            break
        
        frame_sketch = create_sketch(frame)
        cv2.imshow("Video", frame_sketch)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    
def main():
    video = cv2.VideoCapture(0)
    start_video(video)

if __name__ == "__main__":
    main()