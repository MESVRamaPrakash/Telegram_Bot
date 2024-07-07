import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import pyttsx3
import cv2

engine = pyttsx3.init()
def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open webcam")
        return None

    ret, frame = cap.read()
    if ret:
        image_path = 'captured_image.jpg'
        cv2.imwrite(image_path, frame)
        cap.release()
        return image_path
    else:
        cap.release()
        print("Error: Unable to capture image")
        return None

if __name__ == "__main__":
    image_path = capture_image()
    if image_path:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")


        raw_image = Image.open(image_path).convert('RGB')

        question = "who is in the picture?"
        print(question)
        engine.say(f"The question is, {question}")
        engine.runAndWait()

        inputs = processor(raw_image, question, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        output = processor.decode(out[0], skip_special_tokens=True)

        print(f"The Answer is, {output}")
        engine.say(f"The Answer is, {output}")
        engine.runAndWait()
    else:
        print("Failed to capture image.")
