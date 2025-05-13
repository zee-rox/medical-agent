import os
from src.reports.ImageReport import process_xray_image
from src.reports.TextReport import generate_final_diagnosis
from src.models.LoadLLM import llm

# Initialize variables
text_insights = None
final_diagnosis_history = None
final_combined_diagnosis = None
final_diagnosis_explanation = None
final_followup = None
diagnosis_data = {}

def generate_combined_report(query_text=None, xray_image=None, patient_history=None):
    """
    Generate a combined medical report using text and image data
    
    Args:
        query_text (str): The clinical query text
        xray_image (str): Path to X-ray image, if any
        patient_history (str): Patient history text if available
    """
    global text_insights, final_diagnosis_history, final_combined_diagnosis
    global final_diagnosis_explanation, final_followup, diagnosis_data
    
    # Initialize final_diagnosis_image with default value
    final_diagnosis_image = "No image provided."
    
    # Process X-ray image if provided
    if xray_image:
        try:
            _, final_diagnosis_image = process_xray_image(xray_image)
            print(f"X-ray processed successfully: {xray_image}")
        except Exception as e:
            print(f"Error processing X-ray image: {str(e)}")
            final_diagnosis_image = "Error processing X-ray image."
    
    # Generate text insights and diagnosis if we have a query
    if query_text:
        try:
            final_diagnosis_history = generate_final_diagnosis(query_text, patient_history)
            print("Generated final diagnosis from text and history")
        except Exception as e:
            print(f"Error generating diagnosis: {str(e)}")
            final_diagnosis_history = "Error generating diagnosis."
    
    # Create combined diagnosis prompt
    combined_diagnosis_prompt = f"""
You are a clinical diagnostic assistant.
Given the following diagnoses:
1. Image Diagnosis:
{final_diagnosis_image}

2. Final Diagnosis with Query + History:
{final_diagnosis_history}

Weight the contributions approximately as follows: 60% from image diagnosis (if available), 20% from the preliminary text diagnosis, and 20% from the integrated history.
Provide a fully integrated final diagnosis report.
Write plainy and clearly. Do NOT incude symbols in the report.
"""
    final_combined_diagnosis_message = [{"role": "user", "content": combined_diagnosis_prompt}]
    final_combined_diagnosis = llm._call(final_combined_diagnosis_message)
    print("\n✅ Final Combined Diagnosis (Image + Query+History):")
    print(final_combined_diagnosis)

    diagnosis_report = f"""
===============================
🏥 Final Diagnosis Report
===============================
1. Final Image Diagnosis:
{final_diagnosis_image}

2. Final Text Diagnosis (Preliminary):
{text_insights}

3. Final Diagnosis with Query + History:
{final_diagnosis_history}

--------------------------------
Combined Final Diagnosis:
{final_combined_diagnosis}
===============================
""".strip()
    print("\n📄 Diagnosis Report:")
    print(diagnosis_report)

    # Export diagnosis data to JSON file
    diagnosis_data.update({
        "image_diagnosis": final_diagnosis_image,
        "diagnosis_with_history": final_diagnosis_history,
        "combined_final_diagnosis": final_combined_diagnosis,
    })
    
    return diagnosis_report