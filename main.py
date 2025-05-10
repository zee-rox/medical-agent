from src.agent.Core import runnable, AgentState, save_graph_visualization, graph
from src.reports.TextReport import get_user_query
from src.imaging.DetectXRAY import detect_chest_xray
from src.utils.DiagnosisExporter import DiagnosisExporter
from src.reports.CombinedReport import diagnosis_report, diagnosis_data
import warnings
import os
import datetime
import datetime

warnings.filterwarnings("ignore")


def main():
    print("🏥 Starting Enhanced Medical Agent Application")
    
    # Create visualizations directory if it doesn't exist
    visualizations_dir = os.path.join(os.getcwd(), "./data/visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Get user query
    user_query = get_user_query()
    
    # Ask about X-ray image
    xray_image = input("\n🖼️ Enter chest X-ray image file path (or press Enter to skip): ").strip()
    xray_results = None
    
    if xray_image:
        try:
            # Process image and detect chest X-ray findings
            xray_results = detect_chest_xray(xray_image)
            xray_context = "Chest X-ray Findings: " + ", ".join(xray_results)
            print("\n🔍 Chest X-ray Detection Results:")
            print(xray_context)
            
            # Add X-ray context to user query
            user_query = f"{user_query}\n\nThe chest X-ray shows the following findings: {', '.join(xray_results)}."
        except Exception as e:
            print(f"\n⚠️ Error processing X-ray image: {str(e)}")
            print("Continuing without X-ray data...")
    
    # Initialize agent state
    agent_state = AgentState(
        input=user_query,
        chat_history=[],
        intermediate_steps=[],
        output={},
        plan=None,
        reflection=None,
        agent_role="planner",
        agent_outcome=None
    )
    
    print("\n🚀 Initiating diagnostic workflow...")
    
    # Save graph visualization
    # save_graph_visualization(runnable)
    
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
            
            # Add combined report data if available
            try:
                from src.reports.CombinedReport import diagnosis_data as combined_diagnosis_data
                diagnosis_export_data.update(combined_diagnosis_data)
            except (ImportError, AttributeError) as e:
                print(f"\n⚠️ Could not import combined diagnosis data: {str(e)}")
            
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