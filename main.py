from src.agent.Core import AgentState, build_graph, save_graph_visualization
from src.reports.TextReport import get_user_query, process_query
from langgraph.checkpoint.memory import MemorySaver
from src.utils.DiagnosisExporter import DiagnosisExporter
from src.reports.CombinedReport import generate_combined_report
import warnings
import os
import datetime

warnings.filterwarnings("ignore")

def build_graph_runnable():
    # Compile the graph with persistent memory
    graph = build_graph()

    try:
        runnable = graph.compile(checkpointer=MemorySaver())
        print("✅ Enhanced graph compiled with custom memory (persistent state).")
        return runnable
    except Exception as e:
        print(f"⚠ Failed to compile graph with custom memory: {str(e)}")
        print("⚠ Compiling without persistence.")
        runnable = graph.compile()
        return runnable


def main():
    print("🏥 Starting Enhanced Medical Agent Application")
    
    runnable = build_graph_runnable()
    # Save graph visualization
    save_graph_visualization(runnable)
    
    # Get user query
    user_query = get_user_query()
    
    # Process the query (to save it in the global state of TextReport)
    process_query(user_query)
    
    # Get patient history ID first (this happens in generate_final_diagnosis)
    # We're calling this separately to control the flow
    from src.reports.TextReport import get_patient_history
    patient_history_text, patient_history_available = get_patient_history()
    
    # Now ask about X-ray image
    from src.reports.ImageReport import get_xray_input
    xray_image = get_xray_input()
    
    if xray_image:
        try:
            # Generate the combined report and display it
            print("\n📋 Generating comprehensive clinical report...")
            combined_report = generate_combined_report(user_query, xray_image, patient_history_text)
            print(f"\n📄 Combined Report Summary Length: {len(combined_report)} characters")

        except Exception as e:
            print(f"\n⚠️ Error processing X-ray image: {str(e)}")
            print("Continuing without X-ray data...")
    
    # Initialize agent state
    agent_state = AgentState(
        input=combined_report,
        chat_history=[],
        intermediate_steps=[],
        output={},
        plan=None,
        reflection=None,
        agent_role="planner",
        agent_outcome=None
    )
    
    print("\n🚀 Initiating diagnostic workflow on the report...")

    
    # Run agent
    graph_output = runnable.invoke(agent_state,
                                   {
                                       "configurable": {"thread_id": "thread-1"}
                                       })
    
    # Check if execution was successful
    if graph_output and "output" in graph_output and graph_output["output"]:
        print("\n✅ Final Diagnosis Report:")
        diagnosis = graph_output.get("output", {}).get("answer", "No diagnosis available")
        print(diagnosis)
        
        # Display reflection if available
        reflection = graph_output.get("reflection", None)
        if reflection and reflection.get("metrics", {}).get("overall_score", 0) > 0:
            print("\n📊 Diagnostic Quality Assessment:")
            metrics = reflection.get("metrics", {})
            print(f"Overall Quality Score: {metrics.get('overall_score', 0):.2f}/1.0")
            
            # Show if refinement was performed
            if reflection.get("needs_refinement") is False:
                print("✓ The diagnosis met quality standards")
            else:
                print("⚠ The diagnosis required refinement to meet quality standards")
        
        # Save the combined diagnosis report
        try:
            # Create diagnosis data for export
            diagnosis_export_data = {
                "graph_diagnosis": diagnosis,
                "quality_score": metrics.get("overall_score", 0) if reflection else 0,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # # Generate combined report
            try:
                
                # Get diagnosis data from the combined report module
                from src.reports.CombinedReport import diagnosis_data as combined_diagnosis_data
                diagnosis_export_data.update(combined_diagnosis_data)
            except (ImportError, AttributeError) as e:
                print(f"\n⚠️ Could not generate combined diagnosis report: {str(e)}")
            
            # Export to JSON
            export_path = DiagnosisExporter.export_to_json(
                diagnosis_data=diagnosis_export_data,
                output_dir=os.getcwd()
                )
            
            print(f"\n💾 Combined diagnosis saved to: {export_path}")
        except Exception as e:
            print(f"\n⚠️ Error saving combined diagnosis: {str(e)}")
    else:
        print("\n❌ The diagnostic process did not complete successfully")
    
    print("\n🏥 Medical Agent completed!")

if __name__ == "__main__":
    main()