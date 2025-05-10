from src.imaging.DetectXRAY import detect_chest_xray
from src.retrieval.Search import search
from src.models.LoadLLM import llm

xray_image = input("\n🖼️ Enter chest X‑ray image file path (or press Enter to skip): ").strip()
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
    
    # Extract key insights from image context
    image_insights_prompt = f"""
You are a clinical data assistant.
Using the following chest X‑ray context from RAG results, summarize the key findings:
{image_context}
"""
    image_insight_message = [{"role": "user", "content": image_insights_prompt}]
    image_insights = llm._call(image_insight_message)
    print("\n✅ Extracted Image Insights:")
    print(image_insights)
    
    # Generate final image diagnosis
    image_diagnosis_prompt = f"""
You are a clinical diagnostic assistant.
Based on the following image insights:
{image_insights}
and the chest X‑ray findings:
{xray_context}
Provide a concise and clear final diagnosis for the chest X‑ray.
"""
    image_diagnosis_message = [{"role": "user", "content": image_diagnosis_prompt}]
    final_diagnosis_image = llm._call(image_diagnosis_message)
    print("\n✅ Final Image Diagnosis:")
    print(final_diagnosis_image)
else:
    xray_context = ""
    final_diagnosis_image = "No image provided."