import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import transforms
from model import PneumoniaModel

def doctor_search(doctor_type):
    """
    Searches the specialist near you
    """
    base_url = "https://www.google.com/search?q="
    query = "{}+near+me".format("+".join(doctor_type.split(" ")))
    final_url = base_url + query

    return final_url


def preprocess_image(image):
    transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224), 
                                transforms.RandomHorizontalFlip(), 
                                transforms.RandomRotation(10), 
                                transforms.RandomAffine(translate=(0.05, 0.05), degrees=0),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])
    image = Image.open(image)
    image = transform(image).float()
    return image
          


def predict(image):
    with torch.no_grad():

        image = preprocess_image(image)
        image = image.unsqueeze(0)
        output = model(image)
        softmax = nn.Softmax(dim=1)
        output = softmax(output)
        pred = torch.argmax(output)
        print(output)
        confidence = torch.max(output).numpy()
        print(confidence * 100)
    return pred, confidence

def app():
    st.title('Pneumonia Detection(Project by Toufiq Rahatwilkar)')
    st.markdown('This app detects if the patient has viral, bacterial or no Pneumonia')
    st.markdown('The app is based on ResNet-50 model pre-trained on ImageNet dataset.')
    st.markdown("#")
    uploaded_image = st.file_uploader('Upload an image to predict')
    

    if uploaded_image:
        st.image(uploaded_image)

        pred_button = st.button("Predict")
        if pred_button:
            prediction, confidence = predict(uploaded_image)
            print(prediction)
            if prediction == 0:
                st.subheader('The patient is not suffering from Pneumonia 😄🎉🎉')
                st.subheader(f'Confidence of model: {confidence*100:.2f}%')
                st.balloons()
                
            elif prediction == 1:
                st.subheader('The patient is suffering from Bacterial Pneumonia 😔')
                st.subheader(f'Confidence of model: {confidence*100:.2f}%')

            elif prediction == 2:
                st.subheader('The patient is suffering from Viral Pneumonia 😔')
                st.subheader(f'Confidence of model: {confidence*100:.2f}%')
            
            if prediction != 0:
                st.markdown("---")
                st.subheader("Specialists 👨‍⚕")
                st.write("Click on the specialist's name to find out the nearest specialist to you ...")
                for s in ['Primary Care Doctor','Lung Specialist'] :
                    doctor = f"- [{s}]({doctor_search(s)})"
                    st.markdown(doctor, unsafe_allow_html=True)
                st.markdown("---")
            

if __name__ == "__main__":
    model = PneumoniaModel()
    model.load_state_dict(torch.load('./resnet50_model.pth', map_location="cpu"))
    model.eval()

    app()

