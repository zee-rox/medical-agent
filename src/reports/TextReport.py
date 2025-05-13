from src.retrieval.Search import (
    search,
    clean_query, 
    compute_query_embedding
)
from src.models.LoadLLM import llm
from src.retrieval.DataLoader import df_history

def get_user_query():
    """Ask for and validate user's clinical query"""
    user_query = input("\n📝 Enter your clinical query: ").strip()
    if not user_query:
        raise ValueError("❌ Query cannot be empty!")
    return user_query

# Initialize these variables when module is imported
user_query = None
cleaned_query = None
query_embedding = None

# These will be set when process_query is called
def process_query(query_text):
    """Process a user query and prepare it for search"""
    global user_query, cleaned_query, query_embedding
    user_query = query_text
    cleaned_query = clean_query(user_query)
    query_embedding = compute_query_embedding(cleaned_query)

# Global variables for patient history
patient_history_text = ""
patient_history_available = False

def get_patient_history():
    """Get patient history if provided"""
    global patient_history_text, patient_history_available
    
    # If we already have patient history, return it
    if patient_history_available:
        return patient_history_text, patient_history_available
    
    patient_history_text = ""
    patient_history_available = False
    patient_id = input("\n🔍 Enter Patient ID (for history integration, or press Enter to skip): ").strip()
    if patient_id:
        try:
            patient_rows = df_history[df_history["subject_id"] == int(patient_id)]
            if patient_rows.empty:
                print(f"⚠ No history found for patient {patient_id}.")
            else:
                # Get the patient history text
                patient_history_text = str(patient_rows["combined_text"].iloc[0])
                patient_history_available = True
                print("✅ Patient history retrieved from history FAISS index.")
        except Exception as e:
            print(f"⚠ Failed to retrieve history for patient {patient_id}: {e}")
            
    return patient_history_text, patient_history_available

def generate_final_diagnosis(query_text=None, patient_history_text=None):
    """Generate final diagnosis incorporating patient history if available"""
    global user_query, text_insights, clinical_context_query, final_diagnosis_history
    
    print(f"📋 Starting diagnosis generation with query: '{query_text}'")
    print(f"📋 Patient history available: {'Yes' if patient_history_text else 'No'}")
                
    # Process patient history insights if available
    final_diagnosis_history = ""
    
    if patient_history_text:
        try:
            # Get clinical context through search
            print("🔍 Searching for clinical context with patient history...")
            clinical_context_query, _ = search(f"User Query: {query_text}. Patient History: {patient_history_text}")
            
            # Generate insights from patient history
            print("📝 Generating diagnosis with history...")
            diagnosis_prompt_history = f"""
You are a clinical diagnostic assistant.
Using the following information:
- Preliminary Diagnosis from Query: {query_text}
- Patient History: {patient_history_text}
- Minimal Clinical Context: {clinical_context_query}
Provide a final, concise, and evidence‑based diagnosis that synthesizes all the information.
Let's think step by step.
"""

            insights_message = [{"role": "user", "content": diagnosis_prompt_history}]
            # print(insights_message)
            final_diagnosis_history = llm._call(insights_message)
            print("\n✅ Final Diagnosis with Query + History:")
            print(final_diagnosis_history)
        except Exception as e:
            print(f"❌ Error in diagnosis with history: {str(e)}")
            import traceback
            traceback.print_exc()
            final_diagnosis_history = "Could not generate diagnosis with history due to an error."
    
    else:
        
        clinical_context_query, _ = search(f"User Query: {query_text}")
        diagnosis_prompt_history = f"""
You are a clinical diagnostic assistant.
Based on the preliminary diagnosis:
{query_text}
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
        print("\n✅ Final Diagnosis with Query:")
        print(final_diagnosis_history)
        
    return final_diagnosis_history

# Initialize global variables
user_query = None
cleaned_query = None
query_embedding = None
clinical_context_query = None
text_insights = None
final_diagnosis_history = None