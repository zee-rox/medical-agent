from src.retrieval.Search import (
    search,
    clean_query, 
    compute_query_embedding
)
from src.models.LoadLLM import llm
from src.retrieval.DataLoader import df_history, history_faiss_index

user_query = input("\n📝 Enter your clinical query: ").strip()
if not user_query:
    raise ValueError("❌ Query cannot be empty!")
cleaned_query = clean_query(user_query)
query_embedding = compute_query_embedding(cleaned_query)


# Retrieve clinical context for the query using RAG (via ColBERT)
clinical_context_query, _ = search(user_query)
print("\n📖 Retrieved Clinical Context (from search tool):")
# print(clinical_context_query)

text_insights_prompt = f"""
You are a clinical diagnostic assistant.
Focus primarily on the patient's free‑text query to generate a preliminary diagnosis.

Query: {user_query}
GENERAL KNOWLEDGE: {clinical_context_query}

Generate a preliminary diagnosis that relies mainly on the query. Let's think step by step.
"""
print("--------Text Insights Prompt--------")
# print(text_insights_prompt)

insight_message = [{"role": "user", "content": text_insights_prompt}]
text_insights = llm._call(insight_message)

print("\n✅ Preliminary Text Diagnosis Generated")
# print(text_insights)

# 2.2 Integrate Patient History (if provided) to Create a Combined Diagnosis
patient_history_embedding = None
patient_history_text = ""
patient_history_available = False
patient_id = input("\n🔍 Enter Patient ID (for history integration, or press Enter to skip): ").strip()
if patient_id:
    try:
        patient_rows = df_history[df_history["subject_id"] == int(patient_id)]
        if patient_rows.empty:
            print(f"⚠ No history found for patient {patient_id}.")
        else:
            row_index_val = int(patient_rows.index[0])
            # if row_index_val >= history_faiss_index.ntotal:
            #     raise ValueError(f"Index {row_index_val} is out of bounds for the history FAISS index (ntotal={history_faiss_index.ntotal}).")
            # patient_history_embedding = history_faiss_index.reconstruct(row_index_val).reshape(1, -1)
            patient_history_text = str(patient_rows["combined_text"].iloc[0])
            patient_history_available = True
            print("✅ Patient history retrieved from history FAISS index.")
    except Exception as e:
        print(f"⚠ Failed to retrieve history for patient {patient_id}: {e}")

patient_insights = ""
if patient_history_available:
    insights_prompt = f"""
You are a clinical data assistant.
Analyze the following patient history details and extract the key insights relevant to the current symptoms:
{patient_history_text}
Let's think step by step.
"""
    insights_message = [{"role": "user", "content": insights_prompt}]
    # print(insights_message)
    patient_insights = llm._call(insights_message)
    print("\n✅ Extracted Patient History Insights:")
    print(patient_insights)
    
    diagnosis_prompt_history = f"""
You are a clinical diagnostic assistant.
Using the following information:
- Preliminary Diagnosis from Query: {text_insights}
- Patient History Insights: {patient_insights}
- Minimal Clinical Context: {clinical_context_query}
Provide a final, concise, and evidence‑based diagnosis that synthesizes all the information.
Let's think step by step.
"""
else:
    diagnosis_prompt_history = f"""
You are a clinical diagnostic assistant.
Based on the preliminary diagnosis:
{text_insights}
and the minimal clinical context:
{clinical_context_query}
Provide a concise and clear final diagnosis.
Let's think step by step.
"""
diagnosis_message = [{"role": "user", "content": diagnosis_prompt_history}]
final_diagnosis_history = llm._call(diagnosis_message)
if "don't have any information" in final_diagnosis_history.lower() or not final_diagnosis_history.strip():
    final_diagnosis_history = (
        "Based on the current clinical context and patient history, there is insufficient evidence to reach a definitive diagnosis. Further evaluation is recommended."
    )
print("\n✅ Final Diagnosis with Query + History:")
print(final_diagnosis_history)

def get_user_query():
    return user_query