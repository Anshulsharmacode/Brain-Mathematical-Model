import streamlit as st
import requests
from PIL import Image
import base64

def main():
    st.title("Lung Tumor Prediction")

    st.write("Upload a medical image (JPG or PNG) of a lung tumor for classification.")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Button to trigger prediction
        if st.button('Predict Tumor'):
            # Prepare file for the POST request
            files = {'file': ('image.jpg', uploaded_file.getvalue(), 'image/jpeg')}
            
            try:
                # Sending the POST request to the FastAPI endpoint
                response = requests.post('http://localhost:5000/predict-tumor/', files=files)
                response.raise_for_status()  # Raise an error for bad responses
                result = response.json()  # Parse the JSON response
                
                # Display prediction results
                st.subheader("Prediction Result:")
                st.write(f"**Tumor Type:** {result['tumor_type']}")
                st.write(f"**Predicted Label:** {result['predicted_label']}")
                st.write(f"**Tumor Likelihood Score:** {result['tumor_likelihood_score']:.4f}")
                
                # Display image analysis plots
                st.subheader("Image Analysis:")
                
                # Displaying each plot in the results
                for plot_name, plot_data in result['plots'].items():
                    st.image(base64.b64decode(plot_data), caption=plot_name.replace('_', ' ').title(), use_column_width=True)

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while making the request: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
