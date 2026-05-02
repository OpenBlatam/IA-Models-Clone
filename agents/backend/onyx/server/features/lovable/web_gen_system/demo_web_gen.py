from .pipeline import WebGenPipeline
import os
import json

def main():
    pipeline = WebGenPipeline()
    
    # 1. HTML Demo
    print("\n--- HTML Demo ---")
    prompt = "Landing page for an AI coding assistant"
    print(f"Generating HTML for: '{prompt}'...")
    html_content = pipeline.run(prompt, style="modern")
    
    output_file = "demo_output.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content) # type: ignore
    print(f"Saved to {output_file}")

    # 2. Next.js Demo
    print("\n--- Next.js Demo ---")
    prompt = "Create a Next.js project for a blog"
    print(f"Generating Next.js project for: '{prompt}'...")
    nextjs_structure = pipeline.run(prompt, target="nextjs")
    
    if isinstance(nextjs_structure, dict):
        print("Generated Next.js files:")
        for path in nextjs_structure.keys():
            print(f" - {path}")
            
    # 3. Expo Demo
    print("\n--- Expo Demo ---")
    prompt = "Create a mobile app for fitness tracking"
    print(f"Generating Expo project for: '{prompt}'...")
    expo_structure = pipeline.run(prompt, target="expo")
    
    if isinstance(expo_structure, dict):
        print("Generated Expo files:")
        for path in expo_structure.keys():
            print(f" - {path}")

    # 4. Dynamic System Orchestration Demo (Web Track)
    print("\n--- Dynamic System Orchestration (Web Track) ---")
    prompt = "Create a complex web app for project management"
    print(f"Running System Orchestration for: '{prompt}'...")
    agent_structure = pipeline.run(prompt, use_agents=True)
    
    if isinstance(agent_structure, dict):
        print("Agents generated files:")
        for path in agent_structure.keys():
            print(f" - {path}")

    # 5. Dynamic System Orchestration Demo (Mobile Track)
    print("\n--- Dynamic System Orchestration (Mobile Track) ---")
    complex_prompt = "Automate a mobile dashboard with login, dark mode, and typescript"
    print(f"Running System Orchestration for: '{complex_prompt}'...")
    
    # The pipeline now automatically routes this to the Mobile Team
    # It also uses Event Bus and Shared Memory internally
    advanced_structure = pipeline.run(complex_prompt, use_agents=True)
    
    if isinstance(advanced_structure, dict):
        print("Mobile automation complete.")

if __name__ == "__main__":
    main()
