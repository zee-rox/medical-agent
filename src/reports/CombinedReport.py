import os
from src.reports.ImageReport import final_diagnosis_image
from src.reports.TextReport import text_insights, final_diagnosis_history
from src.models.LoadLLM import llm
from src.utils.DiagnosisExporter import DiagnosisExporter

combined_diagnosis_prompt = f"""
You are a clinical diagnostic assistant.
Given the following diagnoses:
1. Image Diagnosis:
{final_diagnosis_image}

2. Preliminary Text Diagnosis (from free‑text query):
{text_insights}

3. Final Diagnosis with Query + History:
{final_diagnosis_history}

Weight the contributions approximately as follows: 60% from image diagnosis (if available), 20% from the preliminary text diagnosis, and 20% from the integrated history.
Provide a fully integrated final diagnosis report.
Let's think step by step.
"""
final_combined_diagnosis_message = [{"role": "user", "content": combined_diagnosis_prompt}]
final_combined_diagnosis = llm._call(final_combined_diagnosis_message)
print("\n✅ Final Combined Diagnosis (Image + Query+History):")
print(final_combined_diagnosis)

# Optional: Extra Explanation and Follow-up Analysis
extra_explanation_prompt = f"""
You are a clinical data assistant.
Provide a brief explanation of the key evidence and reasoning steps that led to the final diagnosis:
{final_combined_diagnosis}
Let's think step by step.
"""
extra_explanation_message = [{"role": "user", "content": extra_explanation_prompt}]
final_diagnosis_explanation = llm._call(extra_explanation_message)
print("\n✅ Extra Diagnosis Explanation:")
print(final_diagnosis_explanation)

followup_prompt = f"""
Based on the final combined diagnosis:

"{final_combined_diagnosis}"

Please summarize the next steps for further clinical evaluation in a concise, evidence‑based manner.
"""
followup_message = [{"role": "user", "content": followup_prompt}]
final_followup = llm._call(followup_message)
print("\n✅ Follow-up Clinical Insights:")
print(final_followup)

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

Explanation of Key Evidence:
{final_diagnosis_explanation}

--------------------------------
Next Steps for Clinical Evaluation:
{final_followup}
===============================
""".strip()
print("\n📄 Diagnosis Report:")
print(diagnosis_report)

# Export diagnosis data to JSON file
diagnosis_data = {
    "image_diagnosis": final_diagnosis_image,
    "text_diagnosis_preliminary": text_insights,
    "diagnosis_with_history": final_diagnosis_history,
    "combined_final_diagnosis": final_combined_diagnosis,
    "key_evidence_explanation": final_diagnosis_explanation,
    "next_steps": final_followup
}

# # Replace the export part at the end with this:
# try:
#     # Export to JSON in data/diagnosis_results folder
#     export_path = DiagnosisExporter.export_to_json(diagnosis_data)
#     print(f"\n💾 Diagnosis saved to: {export_path}")
# except Exception as e:
#     print(f"\n⚠️ Error saving diagnosis: {str(e)}")
#     # Try with an absolute path as fallback
#     try:
#         fallback_path = os.path.join(os.path.expanduser("~"), "medical_diagnosis_reports")
#         export_path = DiagnosisExporter.export_to_json(diagnosis_data, fallback_path)
#         print(f"\n💾 Diagnosis saved to fallback location: {export_path}")
#     except Exception as e2:
#         print(f"\n❌ Failed to save diagnosis report: {str(e2)}")