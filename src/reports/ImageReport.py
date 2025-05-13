from src.imaging.DetectXRAY import detect_chest_xray
from src.retrieval.Search import search
from src.models.LoadLLM import llm

# Initialize global variables
xray_context = ""
image_insights = ""
final_diagnosis_image = "No image provided."

def get_xray_input():
    """
    Prompt user for X-ray image path
    """
    return input("\n🖼️ Enter chest X‑ray image file path (or press Enter to skip): ").strip()

def process_xray_image(xray_image=None):
    """
    Process an X-ray image and generate diagnosis
    
    Args:
        xray_image (str): Path to the X-ray image. If None, will prompt user.
        
    Returns:
        tuple: (xray_context, final_diagnosis)
    """
    global xray_context, image_insights, final_diagnosis_image
    
    if xray_image is None:
        xray_image = get_xray_input()
        
    if xray_image:
        # Process image and detect chest X‑ray findings
        xray_results = detect_chest_xray(xray_image)
        xray_context = "Chest X‑ray Findings: " + ", ".join(xray_results)
        print("\n🔍 Chest X‑ray Detection Results:")
        print(xray_context)
        
        # Retrieve image-specific context via RAG
        image_context, image_rag_chunks = search(xray_context)
        # print("\n📦 Retrieved RAG Chunks for Image:")
        # for idx, chunk in enumerate(image_rag_chunks):
        #     print(f"Chunk {idx+1}: {chunk}\n")
        print("Extracted image context from RAG")
        
        # Extract key insights from image context
        image_insights_prompt = f"""
You are a clinical data assistant.
Using the following chest X‑ray context from RAG results, summarize the key findings. Write plainy and clearly. Do NOT incude symbols in the report.:
{image_context}
"""
        image_insight_message = [{"role": "user", "content": image_insights_prompt}]
        final_diagnosis_image = llm._call(image_insight_message)
        
        print("\n✅ Final Image Diagnosis:")
        print(final_diagnosis_image)
    else:
        xray_context = ""
        final_diagnosis_image = "No image provided."
        
    return xray_context, final_diagnosis_image