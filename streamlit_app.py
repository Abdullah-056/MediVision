import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Medicine Name Recognition",
    page_icon="💊",
    layout="wide"
)

class MedicinePredictor:
    def __init__(self, checkpoint_path, device='cpu'):
        """
        Initialize the predictor with a trained model checkpoint.
        
        Args:
            checkpoint_path: Path to the best_resnet18.pt file
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get label mapping
        self.label_map = checkpoint['label_map']
        self.num_classes = len(self.label_map)
        
        # Initialize model
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)
        
        # Load weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Define transforms (same as validation/test)
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image, top_k=5):
        """
        Predict medicine name from an image.
        
        Args:
            image: PIL Image object
            top_k: Return top K predictions with probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            
            # Get top K predictions
            top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes), dim=1)
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'medicine_name': self.label_map[idx],
                'confidence': float(prob),
                'confidence_percent': f"{prob*100:.2f}%"
            })
        
        return {
            'top_prediction': predictions[0]['medicine_name'],
            'confidence': predictions[0]['confidence'],
            'all_predictions': predictions
        }

@st.cache_resource
def load_model(checkpoint_path):
    """Load the model with caching"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = MedicinePredictor(checkpoint_path, device=device)
    return predictor

def main():
    # Header
    st.title("💊 Medicine Name Recognition System")
    st.markdown("""
    ### Handwritten Medicine Name Recognition using Deep Learning
    
    This application uses a ResNet18-based deep learning model to recognize handwritten medicine names from images.
    Upload an image or multiple images of handwritten medicine names to get predictions.
    """)
    
    # Checkpoint path
    checkpoint_path = "./best_resnet18.pt"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Model checkpoint not found at: {checkpoint_path}")
        st.info("Please ensure the model checkpoint file exists in the specified path.")
        return
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            predictor = load_model(checkpoint_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Display available classes
    st.markdown("---")
    st.subheader("📋 Available Medicine Classes")
    
    # Create searchable dropdown
    classes_list = predictor.label_map
    selected_class = st.selectbox(
        "Search or select a medicine class to see if it's in the model:",
        options=[""] + classes_list,
        index=0
    )
    
    if selected_class:
        class_index = classes_list.index(selected_class)
        st.info(f"✅ '{selected_class}' is available in the model (Class ID: {class_index})")
    
    # Show total number of classes
    st.caption(f"Total classes: {len(classes_list)}")
    
    st.markdown("---")
    
    # Mode selection
    st.subheader("🔍 Prediction Mode")
    prediction_mode = st.radio(
        "Choose prediction mode:",
        ["Single Image", "Batch Images"],
        horizontal=True
    )
    
    if prediction_mode == "Single Image":
        # Single image prediction
        st.markdown("### Single Image Prediction")
        uploaded_file = st.file_uploader(
            "Upload an image of handwritten medicine name",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a single image file"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", width=400)
            
            # Predict button
            if st.button("🔮 Predict", type="primary"):
                with st.spinner("Predicting..."):
                    result = predictor.predict(image, top_k=5)
                
                # Display results
                confidence_percent = result['confidence'] * 100
                top_pred = result['top_prediction']
                
                st.markdown("### Prediction Results")
                
                if confidence_percent >= 50:
                    # High confidence
                    st.markdown(f"**{top_pred}**")
                    st.markdown(f"Confidence: {confidence_percent:.2f}%")
                else:
                    # Low confidence - show confusion
                    st.warning("⚠️ Low Confidence Prediction - I am confused between the following options:")
                    
                    # Show top 2 predictions
                    top2 = result['all_predictions'][:2]
                    st.markdown("**Confused between:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**1. {top2[0]['medicine_name']}**\n\nConfidence: {top2[0]['confidence_percent']}")
                    with col2:
                        st.info(f"**2. {top2[1]['medicine_name']}**\n\nConfidence: {top2[1]['confidence_percent']}")
                
                # Show all top 5 predictions in expander
                with st.expander("View all top 5 predictions"):
                    for i, pred in enumerate(result['all_predictions'], 1):
                        st.write(f"{i}. {pred['medicine_name']} - {pred['confidence_percent']}")
    
    else:
        # Batch image prediction
        st.markdown("### Batch Image Prediction")
        uploaded_files = st.file_uploader(
            "Upload multiple images of handwritten medicine names",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple image files"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} image(s) uploaded")
            
            if st.button("🔮 Predict All", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        image = Image.open(uploaded_file).convert('RGB')
                        result = predictor.predict(image, top_k=5)
                        results.append({
                            'image': image,
                            'filename': uploaded_file.name,
                            'result': result
                        })
                    except Exception as e:
                        results.append({
                            'image': None,
                            'filename': uploaded_file.name,
                            'error': str(e)
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                progress_bar.empty()
                
                # Display results
                st.markdown("### Batch Prediction Results")
                
                for i, item in enumerate(results):
                    if 'error' in item:
                        st.error(f"❌ Error processing {item['filename']}: {item['error']}")
                    else:
                        # Create a container for each result
                        with st.container():
                            # Display image
                            st.image(item['image'], caption=item['filename'], width=400)
                            
                            # Display prediction
                            result = item['result']
                            confidence_percent = result['confidence'] * 100
                            top_pred = result['top_prediction']
                            
                            st.markdown(f"**{top_pred}**")
                            st.markdown(f"Confidence: {confidence_percent:.2f}%")
                            
                            # Show top predictions in expander
                            with st.expander(f"🔍 View all predictions for {item['filename']}"):
                                for j, pred in enumerate(result['all_predictions'], 1):
                                    st.write(f"{j}. **{pred['medicine_name']}** - {pred['confidence_percent']}")
                    
                    # Add separator line (except for last item)
                    if i < len(results) - 1:
                        st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Medicine Name Recognition System | Powered by ResNet18</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

